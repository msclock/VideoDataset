import fast_context_queue.queue as fq

q = fq.Queue()
t2 = (1, 2, 3)
q.put(t2)
assert q.get() == t2

# t1 = (1, 2)
# q.put(t1)
# assert q.get() == t1
print("test_fast_queue_tuple")
