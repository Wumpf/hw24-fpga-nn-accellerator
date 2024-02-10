def write_memh(data, filename):
    with open("weights.memh", "w") as f:
        for m in memory:
            f.write(f"{m:0{4}x} ")


size_in_bytes = 256

memory = [i for i in range(size_in_bytes)]
print("sum: ", sum(memory))

write_memh(memory, "weights.memh")
write_memh(memory, "activations.memh")
