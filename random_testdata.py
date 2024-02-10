def write_memh(data, filename):
    with open(filename, "w") as f:
        for m in data:
            f.write(f"{m:0{4}x} ")


size_in_bytes = 256 * 2

weights = [i for i in range(size_in_bytes)]
activations = [i for i in range(size_in_bytes)]

accumulator = 0
for w, a in zip(weights, activations):
    accumulator += w * a
print(accumulator)

write_memh(weights, "weights.memh")
write_memh(activations, "activations.memh")
