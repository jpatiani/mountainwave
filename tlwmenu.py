from tkinter import *
import numpy as np
import matplotlib
matplotlib.use("TkAgg") # Penting untuk dilakukan, jika tidak program
                        # akan crashed. Pemanggilan ini harus
                        # dilakukan sebelum import pyplot
import matplotlib.pyplot as p
"""
a=1000*get(h_half,'value');   
H=1000*get(h_H,'value');   
ho=1000*get(h_maxht,'value');           
U=get(h_wndspd,'value');     
Lupper=.0001*get(h_Lupper,'value');     
Llower=.0001*get(h_Llower,'value');    
xdom=1000*get(h_xwidth,'value');   
zdom=1000*get(h_vert,'value');    
mink=get(h_mink,'value')/a;   
maxk=get(h_maxk,'value')/a;

Calculate Scorer and Rossby
H=1000*get(h_H,'value');
Llower=.0001*get(h_Llower,'value');  
Lupper=.0001*get(h_Lupper,'value');  
scorer=4*H*H*(Llower*Llower-Lupper*Lupper)/(pi*pi);  
set(h_scorer,'string',num2str(scorer));  

U=get(h_wndspd,'value');
a=1000*get(h_half,'value');
rossby=U/(f*a);
set(h_rossby,'string',num2str(rossby));
"""
def tlwplot(Lupper,Llower,U,H,a,ho,xdom,zdom,mink,maxk):
    """
    Function TLWPLOT
    Called by: TLWMENU
    
    Original by:
            Robert Hart for Meteo 574 / Fall 95
            Penn State University Meteorology
    Porting to python by:
            jpatiani 20170908
            Weather and Climate Prediction Laboratory
        
    Program which analyzes and simulates the flow over
    an isolated mountain in the presence of a two- 
    layer atmosphere.   
        
    Parameters received from menu: 
    -------------------------------
    Lupper - Scorer parameter of upper layer  
    Llower - Scorer parameter of lower layer
    U      - Surface wind speed (in m/s)
    H      - Height above ground of 2-layer interface (in m) 
    a      - Half-width of mountain (in m)  
    ho     - maximum height of mountain 
    xdom   - Horizontal extent of domain (m)
    zdom   - Vertical extent of domain (m)  
    mink   - Minimum wavenumber in Fourier analysis of flow
    maxk   - Maximum wavenumber in Fourier analysis of flow   
    --------------------------------------------------------------
    """
    npts = 40               # cells in each direction
    dk   = 0.367/a          # wavenumber interval size
                            # (smaller the better, but
                            # need to watch calc. time!)
    nk   = (maxk-mink)/dk   # loops for wave# integration
    minx = -.25*xdom        # leftmost limit of domain
    maxx = .75*xdom         # rightmost limit of domain
    minz = 0                # lower limit of domain
    maxz = zdom             # upper limit of domain
    
    matrix1 = np.zeros((npts+1,npts+1))     # temp matrix used in integration
    matrix2 = np.zeros((npts+1,npts+1))     # temp matrix used in integration
    matrix3 = np.zeros((npts+1,npts+1))     # sum matrix used in integration
    
    dx   = (maxx-minx)/npts                 # grid cell size in horizontal
    dz   = (maxz-minz)/npts                 # grid cell size in vertical
    
    x    = np.arange(minx,maxx+dx,dx)       # array of x gridpoints
                                            # np.arange tidak mengikutsertakan stop poin
                                            # sehingga perlu tambahan angka diakhir
                                            # atau bisa juga menggunakan perintah 
                                            # np.linspace dengan jumlah point npts+1
    z    = np.arange(minz,maxz+dz,dz)       # array of z gridpoints
    k    = np.arange(mink,maxk,dk)          # array of wavenumbers tanpa tambahan data
    
    x,z  = np.meshgrid(x,z)                 # mesh arrays into 2-d matrix
    ht   = 0                # initialize weighting for transform
    for kloop in range(0,int(nk)):
        kk = k[kloop]
        m  = np.lib.scimath.sqrt(Llower*Llower-kk*kk) if (Llower*Llower-kk*kk) < 0 else np.sqrt(Llower*Llower-kk*kk)
        n  = np.lib.scimath.sqrt(kk*kk-Lupper*Lupper) if (kk*kk-Lupper*Lupper) < 0 else np.sqrt(kk*kk-Lupper*Lupper)
        if (m+1j*n==0):
            r = 9e99
        else:
            r = (m-1j*n)/(m+1j*n)
            
        R = r*np.exp(2*1j*m*H)
        A = (1+r)*np.exp(H*n+1j*H*m)/(1+R)
        C = 1/(1+R)
        D = R*C
        hs= np.pi*a*ho*np.exp(-a*np.abs(kk))
        ht= ht+np.pi*dk*a*np.exp(-a*np.abs(kk))
               
        aboveH = A*np.exp(-z*n)*(z>H)
        belowH = (C*np.exp(1j*z*m)+D*np.exp(-1j*z*m))*(z<=H)
        
        matrix2 = ((-1j*kk*hs*U*(aboveH+belowH))*np.exp(-1j*x*kk))
        if kloop > 1:
            matrix3 = matrix3+.5*(matrix1+matrix2)*dk
            
        matrix1 = matrix2

    w = np.real(matrix3/ht)
    Hline = H*np.ones((npts+1))
    stream(x,z,U,w,10)
    p.plot(x[0,:],Hline,'m--')
    print(w)
    
    fig = p.figure()
    ax  = fig.add_subplot(111)
    bounds = [-10.,-8.,-6.,-4.,-2.,0.,2.,4.,6.,8.,10.]
    cbounds= np.arange(-10.,10.01,0.2)
    cnf = ax.contourf(x,z,w,cbounds,vmin=-10,vmax=10,
            interpolation=None,cmap='jet', extend="both")
    line= ax.plot(x[0,:],Hline,'m--')
    p.xlabel('X (m)')
    p.ylabel('Height (m)')
    p.title('Vertical Velocity (m/s)')
    p.yticks([0.,1000.,2000.,3000.,4000.,5000.,6000.,7000.,
        8000.,9000.,10000.,])
    cnf.set_clim(-10,10)
    cb = p.colorbar(cnf,ticks=bounds, extend='min')
    p.show()

def stream(x,y,U,v,num):
    """
    Function STREAM
    Called by: TLWPLOT
    
    Original by:
        Robert Hart for Meteo 574 / Fall 95
        Penn State University Meteorology

    Porting to python by:
        jpatiani 20170907
        Weather and Climate Prediction Laboratory

    This subroutine performs a streamline analysis
    of the wind field.
    
    Parameters passed to this routine:
    ----------------------------------
    x   - array of x-gridpoints
    y   - array of y-gridpoints
    U   - speed of x-direction wind (constant)
    v   - array of y-direction wind
    num - # of evenly spaced streamlines to draw
    
    NOTE: Function is written for a constant
    horizontal wind velocity.
    """
    # hold

    xsize = x.shape[0]  # check shape of x
    ysize = y.shape[1]  # size y-grid
    miny  = y[0,0]
    maxy  = y[-1,0]
    minx  = x[0,0]
    maxx  = x[0,-1]
    
    dx    = (maxx-minx)/xsize   # x-dir grid spacing
    dy    = (maxy-miny)/ysize   # y-dir grid spacing
    dh    = ysize/num           # streamline spacing
    
    tstep = dx/U                # time to cross cell
    mtncolor = [.02, .77, .02]   # color of mountain (it green here)
    
    fig = p.figure()
    ycell = 1
    for j in range(0,num):
        ycell = 1+dh*(j)
        if ycell < 0:
            ycell=0
        
        if ycell > ysize:
            ycell = ysize
        
        ax = []
        ay = []
        ax.append(minx)
        ycell = int(round(ycell))
        ycell = ycell - 1
        ay.append(y[ycell,0])
        
        for i in range(1,xsize):
            ax.append(x[ycell,i])
            ay.append(ay[i-1]+tstep*v[ycell,i])
            
        if j == 0:
            ax.append(maxx)
            ay.append(miny)
            p.fill_between(ax,ay,0,color=mtncolor)
        else:
            p.plot(ax,ay,'b-')

        p.grid(True)
        p.ylim(miny,maxy)
        p.xlim(minx,maxx)
        p.yticks([0.,1000.,2000.,3000.,4000.,5000.,6000.,
            7000.,8000.,9000.,10000.])
        p.xlabel('X (m)')
        p.ylabel('Height (m)')
        p.title('Streamline Analysis')
        del(ax)
        del(ay)

def updateWspd(event):
    val = updateRossby()
    rossby.set(val)

def updateHlfwdt(event):
    val = updateRossby()
    rossby.set(val)

def updateIntht(event):
    val = updateScorer()
    scorer.set(val)

def updateLlower(event):
    val = updateScorer()
    scorer.set(val)

def updateLupper(event):
    val = updateScorer()
    scorer.set(val)

def updateScorer():
    H = 1000*intht.get()
    Llower = .0001*llower.get()
    Lupper = .0001*lupper.get()
    scorer = 4*H*H*(Llower*Llower-Lupper*Lupper)/(np.pi*np.pi)
    scorer = "{0:.4f}".format(scorer)
    return scorer

def updateRossby():
    U = wspd.get()
    a = 1000*hlfwdt.get()
    rossby = U/(f*a)
    rossby = "{0:.4f}".format(rossby)
    return rossby

def info():
    about_msg="""
    --------------------------------------------
                   INTRODUCTION
    --------------------------------------------
    This is an interactive model for visualizing
    airflow over an isolated mountain in the
    presence of an atmosphere having varying
    thermodynamical vertical structure. 

    The atmosphere is divided into two layers,
    each having a uniform Scorer parameter.

    A witch of agnesi shape has been chosen
    for the terrain. 

    The resulting airflow over the mountain
    is then calculated as a Fourier sum of the
    individual airflow patterns for each
    wavenumber in the spectral domain.

    The structure of the atmosphere, terrain 
    domain, or wave spectrum can be altered
    by moving the sliders to the requested
    values.  Once ANALYZE FLOW is clicked, a
    contour plot of vertical velocity and an 
    estimate of the corresponding streamlines is
    be plotted, using the profiles set by the
    user (and assuming steady flow).

    BUTTONS:
    INFO - This help screen. 
    ANALYZE FLOW - Calculate and display airflow
                    based on chosen parameters.
    QUIT - Close and exit menu.

    The following is a description of each of
    the sliders on the menu. 

    --------------------------------------------
                ATMOSPHERIC PROFILE:            
    --------------------------------------------
    Sfc Wind: 
        the surface wind speed in m/s
    Lupper:
        Scorer parameter of the upper layer,
        in multiples of 10^-4.
    Llower: 
        Scorer parameter of the lower layer,
        in multiples of 10^-4 
    Interface Height:
        Height above the ground (in km) of the
        interface between the two layers.
    Scorer Condition:
        Condition found by Scorer which must
        be satisfied for the resulting
        airflow to contain trapped waves. 
        Mathematically, 
        4*(H/pi)^2*(Llower^2 -Lupper^2 ) > 1
        where H is the interface height. 
    Rossby Number:
        The rossby number of the flow at 45 deg.
        This model assumes no rotation; 
        therefore, this number is displayed to
        let the user know when the chosen
        combination of wind and domain are such
        that coriolis accelerations make the
        calculated plots questionable. A number
        less than 1 indicates the rotational 
        effects are significant.

    --------------------------------------------
                  TERRAIN PROFILE:              
    --------------------------------------------
    Max Height:
        The maximum height of the isolated
        mountain, in kilometers.
    Half-width:
        The horizontal distance from the center
        of the mountain at which the mountain
        height decreases to one-half the maximum
        height.  (in km).

    --------------------------------------------
                 DOMAIN PROFILE:                
    --------------------------------------------
    Horizontal:
        The width (in km) of the horizontal 
        domain to be analyzed.
    Vertical:
        The height (in km) of the vertical
        domain to be analyzed. 

    --------------------------------------------
                SPECTRAL PROFILE:               
    --------------------------------------------
    Mininum Wavenumber: 
        The smallest wavenumber wave to be
        included in the calculation
        (in multiples of mtn half-widths).
        This is the lower limit on the Fourier
        integration for wavenumber. 
    Maximum Wavenumber: 
        The largest wavenumber wave to be 
        included in the airflow calculation
        (in multiples of mtn half-widths).
        This is the upper limit on the Fourier 
        integration for wavenumber. 
    --------------------------------------------
    """
    top = Toplevel(width=200, height=500)
    top.title("Mountain Lee Wave Model Help/Info")
    
    txt = Text(top, borderwidth=3, relief=RIDGE)
    txt.config(font=("consolas", 12), wrap='word')

    scrollbar = Scrollbar(top, command=txt.yview)
    scrollbar.pack(side=RIGHT, fill=Y)
    
    txt.insert(END, about_msg)
    txt.config(state=DISABLED)
    txt.pack()
    txt['yscrollcommand'] = scrollbar.set
    
    button = Button(top, text="Close", command=top.destroy)
    button.pack()


def analyzeFlow():
    a = 1000*hlfwdt.get()
    H = 1000*intht.get()
    ho= 1000*maxhgt.get()
    U = wspd.get()
    Lupper = .0001*lupper.get()
    Llower = .0001*llower.get()
    xdom   = 1000*hdom.get()
    zdom   = 1000*vdom.get()
    mink   = smink.get()/a
    maxk   = smaxk.get()/a
    tlwplot(Lupper,Llower,U,H,a,ho,xdom,zdom,mink,maxk)

def quit():
    global root
    root.quit()
    root.destroy()

# Define constant
omega=7.292e-5
latit=np.pi/2
f=2*omega*np.sin(latit)

# Start GUI
root = Tk()
root.geometry("600x500+30+30")
menubar = Menu(root)

twspd =Label(root,text='INTERACTIVE MODEL FOR 2-LAYER FLOW OVER AN ISOLATED MOUNTAIN',
        bg='black',fg='white')
twspd.pack(padx=5,pady=10,side=TOP)

"""LEFT PANEL"""
tatmp = Label(root,text='ATMOSPHERIC PROFILE',bg='black',fg='white')
tatmp.place(x=20,y=50,width=270,height=20)

# windspeed
lwspd = Label(root,text='Sfc. Wind (m/s):',bg='black',fg='white')
lwspd.place(x=20,y=98,width=110,height=20)
wspd = Scale(root, from_=0., to=100., orient=HORIZONTAL, resolution=0.5)
wspd.set(20)
wspd.place(x=140,y=80,width=150,height=40)
wspd.bind("<B1-Motion>",updateWspd)
wspd.bind("<ButtonRelease-1>",updateWspd)

# lupper
llupper = Label(root,text='L Upper\n(10^-4):',bg='black',fg='white')
llupper.place(x=20,y=130,width=110,height=40)
lupper = Scale(root,from_=0.,to=50.,orient=HORIZONTAL,resolution=0.5)
lupper.set(4)
lupper.place(x=140,y=120,width=150,height=40)
lupper.bind("<B1-Motion>",updateLupper)
lupper.bind("<ButtonRelease-1>",updateLupper)

# llower
lllower = Label(root,text='L Lower\n(10^-4):',bg='black',fg='white')
lllower.place(x=20,y=170,width=110,height=40)
llower = Scale(root,from_=0.,to=50.,orient=HORIZONTAL,resolution=0.5)
llower.set(10)
llower.place(x=140,y=160,width=150,height=40)
llower.bind("<B1-Motion>",updateLlower)
llower.bind("<ButtonRelease-1>",updateLlower)

# interface ht.
lintht = Label(root,text='Interface Ht.\n(km):',bg='black',fg='white')
lintht.place(x=20,y=230,width=110,height=40)
intht = Scale(root,from_=0.,to=20.,orient=HORIZONTAL,resolution=0.1)
intht.set(3.5)
intht.place(x=140,y=220,width=150,height=40)
intht.bind("<B1-Motion>",updateIntht)
intht.bind("<ButtonRelease-1>",updateIntht)

"""RIGHT PANEL"""
# Terrain profile
tterp = Label(root,text='TERRAIN PROFILE',bg='black',fg='white')
tterp.place(x=310,y=50,width=270,height=20)

# Max height
lmaxhgt = Label(root,text='Max. Height (km)',bg='black',fg='white')
lmaxhgt.place(x=310,y=98,width=110,height=20)
maxhgt = Scale(root,from_=0.,to=3.,orient=HORIZONTAL,
        resolution=0.03)
maxhgt.place(x=430,y=80,width=150,height=40)
maxhgt.set(0.5)

# Half width
lhlfwdt = Label(root,text='Half-width (km)',bg='black',fg='white')
lhlfwdt.place(x=310,y=140,width=110,height=20)
hlfwdt = Scale(root,from_=0.,to=25.,orient=HORIZONTAL,
                resolution=0.25)
hlfwdt.place(x=430,y=120,width=150,height=40)
hlfwdt.set(2.5)
hlfwdt.bind("<B1-Motion>",updateHlfwdt)
hlfwdt.bind("<ButtonRelease-1>",updateHlfwdt)

# Domain profile
tdomp = Label(root,text='DOMAIN PROFILE',bg='black',fg='white')
tdomp.place(x=310,y=180,width=270,height=20)

# Horizontal
lhdom = Label(root,text='Horizontal (km):',bg='black',fg='white')
lhdom.place(x=310,y=230,width=110,height=20)
hdom = Scale(root,from_=0.,to=100.,orient=HORIZONTAL,
                        resolution=1)
hdom.place(x=430,y=210,width=150,height=40)
hdom.set(40)

# Vertical
lvdom = Label(root,text='Vertical (km):',bg='black',fg='white')
lvdom.place(x=310,y=270,width=110,height=20)
vdom = Scale(root,from_=0.,to=20.,orient=HORIZONTAL,
                                resolution=0.2)
vdom.place(x=430,y=250,width=150,height=40)
vdom.set(10)

# Spectral profile
tspec = Label(root,text='SPECTRAL PROFILE',bg='black',fg='white')
tspec.place(x=310,y=300,width=270,height=20)

# Min. Wave#
lsmink = Label(root,text='Min. Wave# \n(half-widths):',bg='black',fg='white')
lsmink.place(x=310,y=340,width=110,height=40)
smink = Scale(root,from_=0.,to=50.,orient=HORIZONTAL,
                                        resolution=0.5)
smink.place(x=430,y=330,width=150,height=40)
smink.set(0)

# Max. Wave#
lsmaxk = Label(root,text='Max. Wave# \n(half-widths):',bg='black',fg='white')
lsmaxk.place(x=310,y=380,width=110,height=40)
smaxk = Scale(root,from_=0.,to=50.,orient=HORIZONTAL,
                                                resolution=0.5)
smaxk.place(x=430,y=370,width=150,height=40)
smaxk.set(30)

# Variable Info
# Scorer label
lscorer = Label(root,text='Scorer Condition:\n(Trapped > 1)',
        bg='black',fg='white')
lscorer.place(x=20,y=310,width=150,height=40)
scorer = StringVar()
scorer.set(updateScorer())
vscorer = Label(root,textvariable=scorer,bg='black',fg='white')
vscorer.place(x=190,y=310,width=100,height=40)

# Rossby label
lrossby = Label(root,text='Rossby Number:\n(45N)',
                bg='black',fg='white')
lrossby.place(x=20,y=380,width=150,height=40)
rossby = StringVar()
rossby.set(updateRossby())
vrossby = Label(root,textvariable=rossby,bg='black',fg='white')
vrossby.place(x=190,y=380,width=100,height=40)

# BOTTOM PANEL
buttonInfo = Button(root, text='Info', command=info)
buttonAnalyze = Button(root, text='Analyze Flow',command=analyzeFlow)
buttonQuit = Button(root, text='Quit', command=quit)
buttonInfo.place(x=150,y=430,width=100,height=40)
buttonAnalyze.place(x=250,y=430,width=100,height=40)
buttonQuit.place(x=350,y=430,width=100,height=40)
root.config(menu=menubar)
root.mainloop()
