from MC_Tktoolbox import *

root = tk.Tk()
root.title("聚合物拉伸行为的格点法模拟")
root.geometry("900x900")
B_载入已有体系= tk.Button(root, text='载入已有体系', command=拉取体系)
B_载入已有体系.pack(side='top', padx=13, pady=3)
B_绘制体系= tk.Button(root, text='绘制体系', command=绘制体系)
B_绘制体系.pack(side='top', padx=13, pady=3)
B_创建新模型 = tk.Button(root,text='创建新模型', command=创建新模型)
B_创建新模型.pack(side='top', padx=13, pady=3)



root.mainloop()