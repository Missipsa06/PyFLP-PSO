from tkinter import *    
from functools import partial
from tkinter.filedialog import askopenfilename
import pandas as pd
import random
import time
import PsoTools as Ptls
import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np
import json
#from PIL import ImageGrab
from numpy import random as rd


trace = 0
class MyEnv(Frame):
    def __init__(self,master):
        Frame.__init__(self,master=None)
        
        self.UNIT = UNIT   # pixels
        self.H = 9# grid height
        self.W = 11 # grid width
        self.x = self.y = 0
        canvas = Canvas(self,height=self.H * UNIT,width=self.W * UNIT,background = 'black', cursor="cross")
#        s = Scrollbar(self, command=canvas.yview)
        canvas.pack(side=LEFT)
#        s.pack(side=RIGHT)
#        canvas.configure(yscrollcommand=s.set)
        self.canvas = canvas
        self.drawn  = None
        self.color = None
        self.bttn_clicks = 0
        self.kinds = [canvas.create_rectangle,canvas.create_oval,canvas.create_line]

        # =============================================================================
        #           Menubar
        # =============================================================================
        self.menubar = Menu(master)
        self.filemenu = Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="New", command= self.NewPlan)
        self.nm = self.filemenu.add_command(label="Open", command= self.Open)
        self.filemenu.add_command(label="Exit", command= master.destroy)
      
        
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        
        
        self.Draw = Menu(self.menubar, tearoff=0)
        self.Add = Menu(self.menubar, tearoff=0)
        self.Add.add_command( label = 'Rectangle', command = self.rectangle )
        self.Add.add_command( label = 'Cercle', command = self.Cercle )
        self.Add.add_command( label = 'Line', command = self.Ligne )
        self.Draw.add_cascade(label="Add", menu=self.Add)
        
        
     
        
        self.Addblock = Menu(self.menubar, tearoff=0)
        self.Draw.add_cascade(label="Add Block", menu=self.Addblock)
        self.menubar.add_cascade(label="Draw", menu=self.Draw)
        
        self.Edit = Menu(self.menubar, tearoff=0)
        self.Edit.add_command(label = 'Move' ,command = self.onMove)
        self.Edit.add_command(label = 'Delete' ,command = self.onDelete)
        self.menubar.add_cascade(label="Edit", menu=self.Edit)
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.label = ['rectangle','cercle','ligne']
        
        self.b1 = Button(self, text="Generate!", command=self.Generate)
        self.b1.pack()
        self.b1.config(bg='dark green', fg='white')
        self.b1.config(font=('helvetica', 20, 'underline italic'))

        with open('Data.json') as data_file:
            self.color = json.load(data_file)

    # =============================================================================
    #         Def functions callback
    # =============================================================================
    # Open new plan  
    def NewPlan(self):
        self.canvas.delete('all')
        Grid = {}
        for c in range(0, self.W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.H * UNIT
            L = self.canvas.create_line(x0, y0, x1, y1,fill = 'white', tags = 'grid',stipple = 'gray50')
            Grid['lines'] = L
            
        for r in range(0, self.H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r,self.W * UNIT, r
            C = self.canvas.create_line(x0, y0, x1, y1,fill = 'white', tags = 'grid',stipple = 'gray50')
            Grid['columns'] = C
        return Grid


    # Open DXF File   
    def Open(self):
        name = askopenfilename(initialdir="D:/Users/BENDJOUDI/Documents/Documents/GithubMaster/Interface/",
                               filetypes =(("CSV File", "*.dxf"),("All Files","*.*")),
                               title = "Choose a file."
                               )

        import dxfgrabber
        dwg = dxfgrabber.readfile(name)
        print("DXF version:{} ".format(dwg.dxfversion))
        
        
        layer_names = dwg.layers.names()
        all_layers = {}
        entitie =  [entity for entity in dwg.entities ]
        
        
        
        for n in layer_names:
            all_layers.update({str(n) : [entity for entity in dwg.entities if (entity.layer == str(n)) and (entity.dxftype == 'LWPOLYLINE' or entity.dxftype == 'SOLID') \
                               and entity.layer != 'CARTOUCHE' and entity.layer != 'S_ZONE___E'  and entity.layer != 'LEGENDE' and entity.layer != 'C_PLA_KR_T']})
    
        layers_list = { i:all_layers[i] for i in all_layers if all_layers[i]!=[] }
        
        for key in layers_list:
            color = []
            for e in layers_list[key]:
                color.append(e.color)
        

        self.NewPlan()
        
        l = {} 
        for key in layers_list:
            f = []
            for e in layers_list[key]:
                k = []
                for i in e.points:
                    k.append((i[0]/300,-i[1]/300))
                f.append(k)
                l.update({key:f})
        i=0
        for key in l:    
            i += 1
            for e in range(len(l[key])):
                
                self.canvas.create_polygon( l[key][e], outline = self.color[i]["hexString"], fill = self.color[i]["hexString"], tags = [str(self.color[i]['colorId']),str(key)])
                print('i = ',i)
            

        return name

    


    # =============================================================================
    #           Drow object with mouse
    # =============================================================================
    def ButtonAdd(self,a,b,c):
        self.canvas.create_polygon(a,outline = b,fill = b)

    def onStart(self, i , event):
        self.shape = self.kinds[i]
        self.start = event
        self.drawn = None
        
    def onGrow(self, event):                         
        canvas = event.widget
        if self.drawn: canvas.delete(self.drawn)
        objectId = self.shape(self.start.x, self.start.y, event.x, event.y, fill = 'gray', tags = 'gray')
        if trace: print(objectId)
        self.drawn = objectId
   
    def rectangle(self):
        self.canvas.bind('<ButtonPress-1>', partial(self.onStart,0))
        self.canvas.bind('<B1-Motion>',     self.onGrow) 

    def Cercle(self):
        self.canvas.bind('<ButtonPress-1>', partial(self.onStart,1))
        self.canvas.bind('<B1-Motion>',     self.onGrow) 
  
    def Ligne(self):
        self.canvas.bind('<ButtonPress-1>', partial(self.onStart,2))
        self.canvas.bind('<B1-Motion>',     self.onGrow) 

    def mouseDown(self, event):
        # remember where the mouse went down
        self.lastx = event.x
        self.lasty = event.y
        
    # =============================================================================
    #               Move objects & detecte overlapping
    # =============================================================================

    def mouseMove(self, event):
        self.canvas.move(CURRENT, event.x - self.lastx, event.y - self.lasty)
        self.lastx = event.x
        self.lasty = event.y
        Current  = self.canvas.find_withtag(CURRENT)
        Id = self.canvas.find_all()
        tg = self.canvas.gettags(Current)
#        print(Id)
#        print(C)
#        print(self.canvas.coords(Current))
        
        
        def tuple_without(original_tuple, element_to_remove):
            new_tuple = []
            for s in list(original_tuple):
                if not s == element_to_remove[0]:
                    new_tuple.append(s)
            return tuple(new_tuple) 
        
        def tuple_with(original_tuple, element_to_remove):
            new_tuple = []
            for s in list(original_tuple):
                if s == element_to_remove[-1]:
                    new_tuple.append(s)
            return tuple(new_tuple) 
        
        Id2 = tuple_without(Id,Current)
        Overlapper = self.canvas.find_overlapping(self.canvas.bbox(CURRENT)[0],self.canvas.bbox(CURRENT)[1],\
                                   self.canvas.bbox(CURRENT)[2],self.canvas.bbox(CURRENT)[3])
        
            
            
        Id = self.canvas.find_all()
        I = ()
        for i in range(15):
            C = self.canvas.find_withtag(self.color[i]['colorId'])
            I = I + C
        Id2 = tuple_without(I,Current)
        self.iden = Id2

        # transforme to couple forme (x,y)
        def trans(p):
            poly = []
            for e in range(len(p[::2])):
                i = 2*e
                poly.append(tuple([p[i],p[i+1]]))
            return poly
        
        p1 = []
        for i in Id2:
            p1.append(trans(self.canvas.coords(i)))
        p2 = trans(self.canvas.coords(CURRENT))
        

        if self.multi_poly(p2,p1):
            self.canvas.itemconfig(Current,fill='red')
        else:
            self.canvas.itemconfig(Current,fill=str(self.color[int(tg[0])]['hexString']))
    
        
    def onMove(self):
        self.canvas.bind('<ButtonPress-1>', self.mouseDown)
        self.canvas.bind('<B1-Motion>', self.mouseMove) 
        self.canvas.update()
        
        
        
    def multi_poly(self,poly1,poly2):
        for i in range(len(poly2)):
            for j in range(len(poly2)):
                if self.polygon_in_polygon(poly1,poly2[i]) and (self.canvas.find_withtag(CURRENT) != self.iden[i]) and self.polygon_in_polygon(poly2[j],poly2[i]):
                    self.canvas.itemconfigure(self.iden[i],fill = self.canvas.gettags(self.iden[i])[0])
                    return True
        return False
    # =============================================================================
    #           Delete objects
    # =============================================================================
    
    def Delete(self, event):
        self.canvas.delete(CURRENT)
        self.start = event
            
    def onDelete(self):    
        self.canvas.bind('<ButtonPress-1>', self.Delete)
        self.canvas.update()
    
    # =============================================================================
    #               Generate FL solution using PSO
    # =============================================================================
    
    
    # Generation Button press
    def Generate(self):
        self.canvas.bind('<ButtonPress-1>', self.PSO(MaxIter = 150,PopSize = 100 ,c1 = 0.7 ,c2 = 1.5,w = 1,wdamp = 0.99))
    
    
    # PSO main
    def PSO(self,MaxIter,PopSize,c1,c2,w,wdamp):
        pop,gbest,model,Vars = Ptls.Init(PopSize)
        with open('Data.json') as data_file:
            color = json.load(data_file)
        
        def makeFig():
            plt.plot(xList,yList) 
        
        plt.ion() # enable interactivity
        fig=plt.figure() # make a figure
        xList=list()
        yList=list()
        for it in range(0, MaxIter):
            for i in range(0, PopSize):
                 
                 for key in pop[i]['velocity'].keys():
                     #Update Velocity for xhat, yhat and rhat
                     pop[i]['velocity'][key] = ( w*pop[i]['velocity'][key]\
                         + c1*np.random.rand(Vars[key]['Size'][0],Vars[key]['Size'][1])*(pop[i]['best']['position'][key] - pop[i]['position'][key]) \
                         + c2*np.random.rand(Vars[key]['Size'][0],Vars[key]['Size'][1])*(gbest['position'][key] - pop[i]['position'][key]))
                     pop[i]['velocity'][key] = pop[i]['velocity'][key].reshape((model['n'],))                    
 
                     
                     
                     #Apply velocity Limits for xhat, yhat and rhat
                     for j in range(0,model['n']):
                         pop[i]['velocity'][key][j] = max(pop[i]['velocity'][key][j],Vars[key]['VelMin'])
                         pop[i]['velocity'][key][j] = min(pop[i]['velocity'][key][j],Vars[key]['VelMax'])
                         
 
                     #Update Position for xhat, yhat and rhat
                     pop[i]['position'][key] = pop[i]['position'][key].reshape((model['n'],))
                     pop[i]['position'][key] = pop[i]['position'][key] + pop[i]['velocity'][key]
 
                     
 
                     for j in range(0,model['n']):
                         #Velocity Mirror Effect for xhat, yhat and rhat
                         if ((pop[i]['position'][key][j] < Vars[key]['Min']) or (pop[i]['position'][key][j] > Vars[key]['Max'])):
                             pop[i]['velocity'][key][j]= - pop[i]['velocity'][key][j]
                             
                         #Apply position limitation for xhat, yhat and rhat    
                         pop[i]['position'][key][j] = max(pop[i]['position'][key][j],Vars[key]['Min'])
                         pop[i]['position'][key][j] = min(pop[i]['position'][key][j],Vars[key]['Max'])
                         
                 # Evaluation        
                 pop[i]['cost'],pop[i]['sol'] = Ptls.CostFunction(pop[i]['position'],model)
                 
                
                    
                 # Apply Mutation
                 NewPop = pop[i]
                 NewPop['position'] = Ptls.Mutate(pop[i]['position'], Vars)
                
                 NewPop['cost'], NewPop['Sol']=Ptls.CostFunction(NewPop['position'],model)
                
                 if NewPop['cost']<pop[i]['cost'] :
                     pop[i]['position'] = NewPop['position'].copy()
                     pop[i]['cost'] = NewPop['cost'].copy()
                     pop[i]['sol'] = NewPop['sol'].copy()
    
            
                        
                # Evaluation 
                 if pop[i]['cost'] < pop[i]['best']['cost']:
                     pop[i]['best']['position'] = pop[i]['position'].copy()
                     pop[i]['best']['cost'] = pop[i]['cost'].copy()
                     pop[i]['best']['sol'] = pop[i]['sol'].copy()
                    
            
                     if pop[i]['best']['cost'] < gbest['cost']:
                         gbest['position'] = pop[i]['best']['position'].copy()
                         gbest['cost']= pop[i]['best']['cost'].copy()
                         gbest['sol']= pop[i]['best']['sol'].copy()
 

            
            # Imporve solution (Apply local search method)  
            NewPop2 = gbest.copy()
#            NewPop2['position'] = Ptls.ImproveSolution(gbest['position'],model,Vars)
            NewPop2['cost'], NewPop2['sol'] = Ptls.CostFunction(gbest['position'],model)
            if NewPop2['cost']<gbest['cost'] :
                gbest['position'] = NewPop2['position'].copy()
                gbest['cost'] = NewPop2['cost'].copy()
                gbest['sol'] = NewPop2['sol'].copy()
            
            if gbest['sol']['v'] == 0:
                Flag = ' (Feasible)'
            else : 
                Flag = ' '
            print('Iteration {}: Best Cost = {}'.format(it+1, gbest['cost']))    
        # =============================================================================
        #     plot solution
        # =============================================================================        
            n = model['n']
            x = gbest['sol']['x']
            y = gbest['sol']['y']
            xl= gbest['sol']['xl']
            yl= gbest['sol']['yl']
            xu= gbest['sol']['xu']
            yu= gbest['sol']['yu']
            xin=gbest['sol']['xin']
            yin=gbest['sol']['yin']
            xout=gbest['sol']['xout']
            yout=gbest['sol']['yout']
            XMin = gbest['sol']['XMIN']
            XMax = gbest['sol']['XMAX']
            YMin = gbest['sol']['YMIN']
            YMax = gbest['sol']['YMAX']
            self.canvas.delete('all')
            self.NewPlan()
            for i in range(0,n):
                self.canvas.create_rectangle(xu[i],yl[i],xl[i],yu[i], fill = color[i*2+1]["hexString"])
                self.canvas.create_oval(xin[i]-10,yin[i]-10,xin[i]+10,yin[i]+10, fill = 'white') 
                self.canvas.create_oval(xout[i]-10,yout[i]-10,xout[i]+10,yout[i]+10, fill = 'green')
                self.canvas.create_rectangle(XMin,YMax,XMax,YMin,outline = 'white', dash=(5, 1, 2, 1))
                self.canvas.create_rectangle(0,model['H'],model['W'],0, outline = 'yellow')
                
                label = self.canvas.create_text(x[i],y[i],text = ''+str(i+1))
                self.canvas.update()
            xList.append([it+1])
            yList.append(gbest['cost'])
            drawnow(makeFig)
            w *= wdamp
        x=self.canvas.winfo_rootx()+self.canvas.winfo_x()
        y=self.canvas.winfo_rooty()+self.canvas.winfo_y()
        x1=x+self.canvas.winfo_width()
        y1=y+self.canvas.winfo_height()
        
#        GenNum = self.update_count() #Get Click number
#        ImageGrab.grab().crop((x,y,x1,y1)).save(GenNum+'.png') # save the solution into PNG File
#        
#        G = {}
#        
#        # Save solution parameters into JSON File
#        for key2 in gbest['sol']:
#            G[key2] = gbest['sol'][key2].tolist()
#        with open(GenNum+'.json', 'w') as fp:
#            json.dump(G, fp)
#        self.b1['text'] = 'Generate again!'

        
    # Count generation button click
    def update_count(self):
        self.bttn_clicks += 1
        self.b1['text'] = str(current_dir)+'/Generation/Sol'+str(self.bttn_clicks)
        return self.b1['text']

# =============================================================================
#               Main()
# =============================================================================

if __name__ == "__main__":
    fen=Tk()
    fen.attributes('-fullscreen', True)
    UNIT = 100
    env = MyEnv(fen)
    env.pack()
    fen.config(menu=env.menubar)
    fen.mainloop()
    