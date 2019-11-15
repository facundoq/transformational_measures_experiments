


n_values=3

def consumer():
    sum =0
    for i in range(n_values):
        val = yield
        print(val)
        sum+=val
    yield None
    yield sum
    return



consumers = [consumer() for i in range(5)]
for c in consumers:
    c.send(None)

for i in range(n_values):
    for c in consumers:
        c.send(i)

values= [next(c) for c in consumers]
print(values)



