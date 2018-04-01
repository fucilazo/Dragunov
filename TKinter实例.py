import tkinter as tk


app = tk.Tk()
app.title('TITLE')

theLablel = tk.Label(app, text = 'This is a program')
theLablel.pack()    # 调整位置

app.mainloop()  # 切到TKinter接管
