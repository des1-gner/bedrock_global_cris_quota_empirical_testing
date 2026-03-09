import os
import boto3
import json
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
OUTPUT_FILE = "empirical_rpm_test_output.txt"

DURATION_S = 60
SMASH_RPM = 1000
GENTLE_RPM = 150

REGIONS = ["us-east-1", "us-east-2", "us-west-2"]

log_lock = threading.Lock()
log_lines = []

def log(msg):
    with log_lock:
        print(msg)
        log_lines.append(msg)

def invoke_model(bedrock, request_id):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}]
    }
    try:
        start = time.time()
        resp = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
        latency = time.time() - start
        result = json.loads(resp['body'].read())
        stop = result.get('stop_reason', '')
        tin = result.get('usage', {}).get('input_tokens', 0)
        tout = result.get('usage', {}).get('output_tokens', 0)
        log(f"[{datetime.now().strftime('%H:%M:%S')}] {request_id}: OK ({latency:.2f}s) stop={stop} in={tin} out={tout}")
        return "SUCCESS"
    except Exception as e:
        log(f"[{datetime.now().strftime('%H:%M:%S')}] {request_id}: ERR - {e}")
        return "ERROR"

def send_stream(executor, bedrock, region, rpm, duration, results):
    interval = 60.0 / rpm
    start = time.time()
    i = 0
    while time.time() - start < duration:
        fut = executor.submit(invoke_model, bedrock, f"{region}-{i}")
        results.append(fut)
        i += 1
        next_send = start + (i * interval)
        wait = next_send - time.time()
        if wait > 0:
            time.sleep(wait)


def test_rpm(smash_region, gentle_regions):
    clients = {}
    clients[smash_region] = boto3.client('bedrock-runtime', region_name=smash_region)
    for r in gentle_regions:
        clients[r] = boto3.client('bedrock-runtime', region_name=r)

    log(f"\nModel: {MODEL_ID}")
    log(f"Smash: {smash_region} @ {SMASH_RPM} RPM")
    for r in gentle_regions:
        log(f"Gentle: {r} @ {GENTLE_RPM} RPM")
    log(f"Duration: {DURATION_S}s")
    log("=" * 70)

    all_futures = {}  # region -> list of futures
    all_futures[smash_region] = []
    for r in gentle_regions:
        all_futures[r] = []

    with ThreadPoolExecutor(max_workers=300) as executor:
        threads = []
        t = threading.Thread(target=send_stream, args=(executor, clients[smash_region], smash_region, SMASH_RPM, DURATION_S, all_futures[smash_region]))
        threads.append(t)
        for r in gentle_regions:
            t = threading.Thread(target=send_stream, args=(executor, clients[r], r, GENTLE_RPM, DURATION_S, all_futures[r]))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        results = {}
        for region, futs in all_futures.items():
            results[region] = [f.result() for f in futs]

    log("\n" + "=" * 70)
    log("RESULTS")
    log("=" * 70)
    for region, res in results.items():
        total = len(res)
        ok = res.count("SUCCESS")
        err = res.count("ERROR")
        log(f"{region}: {total} total | {ok} ok ({ok/total*100:.0f}%) | {err} err ({err/total*100:.0f}%)")

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(log_lines))
    log(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    print("Which region to smash?")
    for i, r in enumerate(REGIONS, 1):
        print(f"{i}. {r}")
    choice = int(input(f"Enter 1-{len(REGIONS)}: ").strip())
    smash = REGIONS[choice - 1]
    gentle = [r for r in REGIONS if r != smash]
    test_rpm(smash, gentle)
