import matplotlib.pyplot as plt
y1=[10,13,5,40,30,60,70,12,55,25]
x1=range(0,10)
x2=range(0,10)
y2=[5,8,0,30,20,40,50,10,40,15]
plt.plot(x1,y1,label='Frist line',linewidth=3,color='r',marker='o',
markerfacecolor='blue',markersize=12)
plt.plot(x2,y2,label='second line')
plt.xlabel('Plot Number')#横向坐标含义
plt.ylabel('Important var')#纵向坐标含义
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()