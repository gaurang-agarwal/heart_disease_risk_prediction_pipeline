import asyncio
import random
import time

import aiohttp

URL = "http://localhost:8000/predict"

# Configuration
CONCURRENT_USERS = 50  # Number of simultaneous requests
TOTAL_REQUESTS = 5000  # Total requests to send

SUCCESS = 0
FAILED = 0


def random_payload():
    """Generate realistic heart disease prediction requests."""
    return {
        "age": random.randint(29, 77),
        "sex": random.randint(0, 1),
        "cp": random.randint(0, 3),
        "trestbps": random.randint(90, 200),
        "chol": random.randint(120, 420),
        "fbs": random.randint(0, 1),
        "restecg": random.randint(0, 2),
        "thalach": random.randint(70, 202),
        "exang": random.randint(0, 1),
        "oldpeak": round(random.uniform(0, 6), 1),
        "slope": random.randint(0, 2),
        "ca": random.randint(0, 4),
        "thal": random.randint(0, 3),
    }


async def send_request(session, request_id):
    global SUCCESS, FAILED

    payload = random_payload()
    start = time.perf_counter()

    try:
        async with session.post(URL, json=payload) as response:
            await response.text()

            latency = (time.perf_counter() - start) * 1000

            if response.status == 200:
                SUCCESS += 1
                print(f"[{request_id}] " f"OK  " f"{latency:.1f} ms")
            else:
                FAILED += 1
                print(f"[{request_id}] " f"FAIL ({response.status}) " f"{latency:.1f} ms")

    except Exception as e:
        FAILED += 1
        print(f"[{request_id}] ERROR: {e}")


async def worker(worker_id, queue):
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        while not queue.empty():
            request_id = await queue.get()
            await send_request(session, request_id)
            queue.task_done()


async def main():
    queue = asyncio.Queue()

    for i in range(TOTAL_REQUESTS):
        queue.put_nowait(i + 1)

    start = time.perf_counter()

    tasks = [asyncio.create_task(worker(i, queue)) for i in range(CONCURRENT_USERS)]

    await queue.join()

    for t in tasks:
        t.cancel()

    elapsed = time.perf_counter() - start

    print("\n========== Load Test Summary ==========")
    print(f"Total Requests : {TOTAL_REQUESTS}")
    print(f"Successful     : {SUCCESS}")
    print(f"Failed         : {FAILED}")
    print(f"Duration       : {elapsed:.2f} sec")
    print(f"Requests/sec   : {TOTAL_REQUESTS / elapsed:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
