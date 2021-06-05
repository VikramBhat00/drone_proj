import matplotlib.pyplot as plt
filename = "t265LogFlight5204"


f = open(filename + ".txt", "r")

lines = f.readlines()

#t = [i for i in range(len(lines))]
t = []
x = []
y = []
z = []
for line in lines:
    
    s = line.split()
    t.append(s[0])
    x.append(s[1])
    y.append(s[2])
    z.append(s[3])

print(x)
t = [float(i) - float(t[0]) for i in t]
plt.plot(t, x, label='x')
plt.plot(t, y, label='y')
plt.plot(t, z, label='z')
plt.xlabel("time (seconds)")
plt.ylabel("distance (meters)")
plt.title(filename)
plt.legend()
plt.show()
