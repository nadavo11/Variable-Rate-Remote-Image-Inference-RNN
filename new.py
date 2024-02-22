import matplotlib
import matplotlib.pyplot as plt

losses = [1,2,3,2,2,3,1]
plt.figure();
plt.plot(losses);
plt.xlabel("loss");
plt.ylabel("epoch")
plt.show()
