#!python3.4
from urllib.request import urlopen
from PIL.ImageGrab import grab
from pymouse import PyMouse
from fractions import gcd
from msvcrt import getch
from io import BytesIO
from os import system
from PIL import Image
import random
import math
import time


class picture:

	def __init__(self, filename, source, scale):
		self.name = filename
		self.source = source
		self.scale = scale

	def load(self):
		rgb = [[0,0,0,0,i,0] for i in range(64)]
		if self.source==1:
			im = Image.open(self.name)
		else:
			fd = urlopen(self.name)
			image_file = BytesIO(fd.read())
			im=Image.open(image_file)
		im = im.convert('RGBA')
		width, height = im.size
		rec = [[-1 for x in range(width)] for y in range(height)]
		pixdata = [[(0,0,0,0) for x in range(width)] for y in range(height)]
		totalpix=0
		print('Loading picture, please wait')
		for y in range(height):
			if y==height//2:
				print('Analyzing picture')
			for x in range(width):
				r,g,b,a = im.getpixel((x,y))
				pixdata[y][x] = r,g,b,a
				if a>128:
					totalpix+=1
					ind = (r//64)*16 + (g//64)*4 + b//64
					rgb[ind][0]+=r
					rgb[ind][1]+=g
					rgb[ind][2]+=b
					rgb[ind][3]+=1
					rec[y][x]=ind
		avecolor=[]
		for i in range(64):
			if rgb[i][3]==0:
				avecolor.append([-1,-1,-1])
			else:
				avecolor.append([rgb[i][0]//rgb[i][3],rgb[i][1]//rgb[i][3],rgb[i][2]//rgb[i][3]])
		srgb = sorted(rgb, key = lambda foo: -foo[3])
		pixcount = 0
		maxcolor=9
		for i in range(9):
			pixcount += srgb[i][3]
			if pixcount > 0.9*totalpix:
				maxcolor=i+1
				break
		while True:
			alist=[]
			los = 1 - pixcount/totalpix
			for ind in range(maxcolor-1):
				alist.append(srgb[ind][4])
			r1,g1,b1,n1=0,0,0,0
			for ind in range(maxcolor-1,64):
				r1+=srgb[ind][0]
				g1+=srgb[ind][1]
				b1+=srgb[ind][2]
				n1+=srgb[ind][3]
			if n1>0:
				r1 = r1//n1
				g1 = g1//n1
				b1 = b1//n1
			im1 = Image.new('RGBA',(width,height))
			pixcount=0
			segcount=0
			print('With {} different colors, recovery rate is {:.2%}'.format(maxcolor, 1-los))
			for y in range(height):
				if y==height//2:
					print('The output picutre will look like this:')
				oldc=-1
				for x in range(width):
					if rec[y][x]==-1:
						im1.putpixel((x,y),(255,255,255,0))
						newc=rec[y][x]
					elif rec[y][x] in alist:
						r,g,b = avecolor[rec[y][x]]
						im1.putpixel((x,y),(r,g,b,255))
						newc=rec[y][x]
					else:
						im1.putpixel((x,y),(r1,g1,b1,255))
						newc=100
					if newc!=oldc:
						if oldc!=-1:
							segcount+=1
						oldc=newc
				if oldc!=-1:
					segcount+=1
			im1.show()
			estm = segcount//500 + (width*height*self.scale)//200000
			print('Estimated painting time is {} minutes.'.format(estm))
			print('Improve output image quanlity (y/n)?')
			while True:
				c=getch().decode()
				if c=='n' or c=='y':
					break
			if c=='n':
				break
			pixcount=0
			for i in range(64):
				pixcount+=srgb[i][3]
				if pixcount > totalpix*(1-los/2):
					break
			maxcolor = i+1
		print('Ready to paint, press any key to start.')
		c=getch().decode()
		if c=='e':
			input()
		palette=[]
		for i in range(maxcolor-1):
			n = srgb[i][4]
			r,g,b = avecolor[n]
			palette.append((n,r,g,b))
		palette.append((-1,r1,g1,b1))
		return pixdata, palette

	def crop(self, pixdata, start_x, start_y, end_x, end_y):
		pixblock = [[(255,255,255,255) for x in range(end_x-start_x)] for y in range(end_y-start_y)]
		for y in range(start_y,end_y):
			for x in range(start_x,end_x):
				pixblock[y-start_y][x-start_x] = pixdata[y][x]
		return pixblock
		
	def parse(self, pixblock, palette, ind):
		segments=[]
		height=len(pixblock)
		width=len(pixblock[0])
		num,red,green,blue=palette[ind]
		for j in range(height):
			flag = 0
			for i in range(width):
				r,g,b,a = pixblock[j][i]
				if (num<0 or (r//64)*16+(g//64)*4+(b//64)==num) and a>150:
					pixblock[j][i] = (0,0,0,0)
					if flag==0:
						flag = 1
						xl = i
				else:
					if flag==1:
						flag=0
						xr = i-1
						segments.append((j,xl,xr))
			if flag==1:
				xr=i
				segments.append((j,xl,xr))
		return segments, pixblock


class paint:	
	
	def __init__(self, mouse):
		self.mouse = mouse
		self.scr_width, self.scr_height = mouse.screen_size()
		self.center_x = 170 + self.scr_width//2
		self.center_y = self.scr_height//2
		if self.scr_width==1280 or self.scr_width==1366:
			self.wheel_x = 502 + self.scr_width//2
			self.wheel_y = 212 + self.scr_height//2
			self.radius = 31
		else:
			self.wheel_x, self.wheel_y, self.radius = 0,0,0
	
	def shift(self, dir, t):
		if dir == 'up':
			self.mouse.move(self.center_x, self.center_y - 318)
		elif dir == 'down':
			self.mouse.move(self.center_x, self.center_y + 318)
		elif dir == 'left':
			self.mouse.move(self.center_x - 318, self.center_y)
		elif dir == 'right':
			self.mouse.move(self.center_x + 318, self.center_y)
		time.sleep(t)
		self.mouse.move(self.center_x, self.center_y)

	def drift(self, dir, t, x0, y0, vec1, vec2):
		self.shift(dir,t)
		time.sleep(3)
		self.mouse.move(1,100)
		im = grab()
		data = [sum(im.getpixel((x0+k*vec1, y0+k*vec2))[:3]) for k in range(409)]
		for k in range(401):
			if data[k] < 100:
				if max(data[k+1:k+3])>600 and min(data[k+2:k+5])<150 and max(data[k+3:k+6])>600 and min(data[k+4:k+7])<150:
					return k
		return -1
		
	def setmouse(self, x, y):
		while True:
			c=getch().decode()
			if c=='a':
				x-=1
			elif c=='d':
				x+=1
			elif c=='w':
				y-=1
			elif c=='s':
				y+=1
			elif c=='j':
				self.shift('left',1)
			elif c=='l':
				self.shift('right',1)
			elif c=='i':
				self.shift('up',1)
			elif c=='k':
				self.shift('down',1)
			elif c=='\r' or c=='\n':
				break
			self.mouse.move(x,y)
		return x,y
	
	def setscr(self):
		print('I need more information about your screen size.')
		print('Please use asdw keys to put the mouse near the center of the color wheel, then press enter.')
		x,y = self.setmouse(502 + self.scr_width//2,212 + self.scr_height//2)
		self.mouse.click(x,y-15)
		time.sleep(0.1)
		im=grab()
		maxr, maxj, cx, lv = 0, 0, 0, 0
		for j in range(-8,9):
			data=[sum(im.getpixel((x-50+k,y+j))) for k in range(100)]
			left, right = -1,0
			for k in range(100):
				if data[k]<60:
					if left<0:
						left=k
					elif k-left>10:
						right=k
						break
			radius = right-left
			if radius>maxr:
				maxr=radius
				maxj = j
				cx=(left+right+1)//2
			elif radius==maxr:
				lv+=1
		x += cx-50
		y += maxj+lv//2
		self.mouse.click(x+15,y)
		time.sleep(0.1)
		im=grab()
		data=[sum(im.getpixel((x,y+j-40))) for j in range(81)]
		top,bottom=-1,-1
		for k in range(40):
			if data[40-k]<300:
				if top<0:
					top=k
			if data[40+k]<200:
				if bottom<0:
					bottom=k
		y = y+(bottom-top)//2
		self.mouse.click(x,y)
		self.wheel_x = x
		self.wheel_y = y
		self.radius = (top+bottom)//2

	def setcolor(self, r, g, b, rainbow=0):
		if self.wheel_x == 0:
			self.setscr()
		wheel_x = self.wheel_x
		wheel_y = self.wheel_y
		wheel_r = self.radius
		bar_x = self.wheel_x+wheel_r+15
		if rainbow == 1:
			xt = wheel_x + int(wheel_r*math.cos(2*math.pi*r))
			yt = wheel_y - int(wheel_r*math.sin(2*math.pi*r))
			self.mouse.click(xt,yt)
			self.mouse.click(bar_x,wheel_y)
			return
		r,g,b = r/255, g/255, b/255
		maxc = max(r, g, b)
		minc = min(r, g, b)
		z = (maxc+minc)/2
		if minc == maxc:
			x,y,xn,yn = 0,0,0,0
		else:
			sc = (maxc-minc)/math.sqrt(r*r+g*g+b*b-r*g-g*b-b*r)
			x = (r - g/2 - b/2) * sc
			y = (math.sqrt(3)/2) * (g-b) * sc
			rd = math.sqrt(x*x+y*y)
			rn = math.sqrt(rd)
			xn = x/rn
			yn = y/rn
		xt = wheel_x + int(wheel_r*xn)
		yt = wheel_y - int(wheel_r*yn)
		zt = wheel_y + wheel_r - round(2*wheel_r*z)
		self.mouse.click(xt,yt)
		self.mouse.click(bar_x,zt)

	def drawline(self, startx, starty, endx, endy):
		self.mouse.press(startx, starty)
		self.mouse.drag(endx, endy)
		time.sleep(0.1)
		self.mouse.release(endx, endy)
		if abs(endx-startx) > 40 or abs(endy-starty)>40:
			self.mouse.click(endx,endy)
			time.sleep(0.1)

	def barcode(self, startx, starty, dirx, diry, norx, nory, clean=False, color=(0,0,0)):
		barnum = 10
		barlen = 14
		barpos=-2
		if clean == False:
			im=grab()
			r,g,b=im.getpixel((startx,starty))
			barnum = 9
			barlen = 12
			barpos=0
		for i in range(barnum):
			if clean==True:
				r,g,b=color
				self.setcolor(r,g,b)
			else:
				col = 255*((i%4)//2)
				self.setcolor(col,col,col)
			self.drawline(startx+norx*i+dirx*barpos, starty+nory*i+diry*barpos, startx+norx*i+dirx*barlen, starty+nory*i+diry*barlen)
			time.sleep(0.1)
		return r,g,b
		
	def drawblock(self, segments, startx, starty, red, green, blue, scale=2):
		if len(segments)>0:
			self.setcolor(red,green,blue)
		for seg in segments:
			y,xl,xr = seg
			self.drawline(startx+scale*xl, starty+scale*y, startx+scale*xr, starty+scale*y)

	def relocate(self, x, y, type):
		print('Use asdw to move the mouse to where the barcode should be.')
		x0,y0 = self.setmouse(x,y)
		if type==0:
			self.barcode(x0,y0,0,1,1,0)
		elif type==1:
			self.barcode(x0,y0,1,0,0,1)
			self.barcode(x0,y0,0,-1,-1,0)
		return x0,y0

	def autodraw(self, filename, source, scale=2):
		pic = picture(filename, source, scale)
		pixdata,palette = pic.load()
		height = len(pixdata)
		width = len(pixdata[0])
		cnum = len(palette)
		xr, yr = 0, 0
		ulx, uly = 0, 0
		safec = 5
		print('Use asdw to push you mouse to the start point')
		xr, yr = self.setmouse(self.center_x-200,self.center_y-200)
		self.setcolor(255,255,255)
		time.sleep(1)
		xr = xr-self.center_x+200
		yr = yr-self.center_y+200
		blkcount=0
		while True:
			ulx=0
			shiftcount = 0
			while True:
				blkcount+=1
				if blkcount==15:
					input()
				shiftcount += 1
				if ulx>0:
					self.barcode(self.center_x+xr+safec-200, self.center_y-5,0,1,1,0, clean=True, color=(r1,g1,b1))
				if ulx+(400-xr)//scale < width:
					self.mouse.move(self.center_x, self.center_y)
					r1,g1,b1 = self.barcode(self.center_x+200+safec,self.center_y-5,0,1,1,0)
				if ulx==0:
					if uly>0:
						self.barcode(self.center_x+xr-safec-201, self.center_y+yr+safec-200,1,0,0,1, clean=True, color=(r2,g2,b2))
						self.barcode(self.center_x+xr-safec-201, self.center_y+yr+safec-200,0,-1,-1,0, clean=True, color=(r2,g2,b2))
					if uly+(400-yr)//scale < height:
						r2,g2,b2 = self.barcode(self.center_x+xr-201-safec, self.center_y+safec+200,1,0,0,1)
						self.barcode(self.center_x+xr-201-safec, self.center_y+safec+200,0,-1,-1,0)
				sx = max(ulx-1,0)
				sy = max(uly-1,0)
				ex = min(1+ulx+(400-xr)//scale, width-1)
				ey = min(1+uly+(400-yr)//scale, height-1)
				pixblock = pic.crop(pixdata,sx,sy,ex,ey)
				for ind in range(cnum):
					seg,pixblock = pic.parse(pixblock, palette, ind)
					red,green,blue = palette[ind][1:]
					self.drawblock(seg, self.center_x+xr-200, self.center_y+yr-200, red, green, blue, scale)
				time.sleep(0.5)
				ulx = ulx + (400-xr)//scale
				if ulx>=width:
					break
				xr = self.drift('right', 1, self.center_x-200, self.center_y, 1, 0)
				if xr<0:
					input('Lost target')
					xt,yt = self.relocate(self.center_x, self.center_y, 0)
					xr = xt - self.center_x + 200
				while xr >= 100:
					xr = self.drift('right', xr/400, self.center_x-200, self.center_y, 1, 0)
					while xr<=0:
						xr = self.drift('left', 0.5, self.center_x-200, self.center_y, 1, 0)
				xr-=safec
			uly = uly + (400-yr)//scale
			if uly>=height:
				break
			xr = self.drift('left', shiftcount/2, self.center_x+199, self.center_y+200, -1, 0)
			cnt = 0
			while xr==-1:
				xr = self.drift('left', 1, self.center_x+199, self.center_y+200, -1,0)
				cnt+=1
				if cnt>=shiftcount+2:
					print('Lost target')
					xt,yt = self.relocate(self.center_x, self.center_y, 1)
					xr = self.center_x + 200 - xt
			while xr<=300:
				xr = self.drift('right', (400-xr)/400, self.center_x+199, self.center_y+200, -1,0)
				while xr<0:
					xr = self.drift('left', 0.5, self.center_x+199, self.center_y+200, -1,0)
			xr = 400-xr
			xr+=safec
			yr = self.drift('down', 1, self.center_x+xr-200, self.center_y-200, 0, 1)
			while yr >= 100:
				yr = self.drift('down', yr/400, self.center_x+xr-200, self.center_y-200, 0, 1)
				while yr<=0:
					yr = self.drift('up', 0.5, self.center_x+xr-200, self.center_y-200, 0, 1)
			yr-=safec

	def plotcurve(self):
		a,b = self.center_x, self.center_y
		print('Choose color:\n 1. rainbow\n 2. monochrome\n 3. random noise\n 4. costomized')
		while True:
			color_plan=getch().decode()
			if color_plan in '1234\n\r':
				break
		if color_plan in '4\n\r':
			color_plan='4'
			while True:
				color_string = input('Input your colors in RGB, separate by comma: ').split(',')
				if len(color_string) != 6:
					print('Please input 6 integers between 0-255')
					continue
				break
			r1,g1,b1,r2,g2,b2 = [int(colrgb)%256 for colrgb in color_string]
			print(r1,g1,b1,r2,g2,b2)
		elif color_plan=='2':
			r1,g1,b1,r2,g2,b2 = 0,0,0,255,255,255
		color_period = input('Color phase (press enter for default): ').split(',')
		cp = [0,1,0,0]
		for i in range(len(color_period)):
			try:
				cp[i]=float(color_period[i])
			except:
				pass
		print('Choose the type of curve')
		print(' 0. spiral(clockwise)\n 1. spiral(counterclockwise)\n 2. circle\n 3. hyperbola-I\n 4. hyperbola-II\n 5. hyperbola-III\n 6. spinograph')
		while True:
			curve_type=getch().decode()
			if curve_type in '0123456':
				break
		if curve_type in '016':
			c=input('Number of loops: ')
			try:
				lpn = float(c)
			except:
				lpn = 1
		if curve_type == '6':
			c=input('Parameter k= ')
			try:
				k = float(c)
			except:
				k = 0.5
			c=input('Parameter l= ')
			try:
				l = float(c)
			except:
				l = 1
		interval = input('Input an interval, press enter for default: [0,1]').split(',')
		try:
			left = float(interval[0])
		except:
			left=0
		try:
			right = float(interval[1])
		except:
			right=1
		num_c = input('Number of curves: ')
		try:
			num=int(num_c)
		except:
			num=1
		for n in range(num):
			ratio = n/num
			t = left + ratio*(right-left)
			col_ind = cp[0]+cp[1]*ratio+cp[2]*ratio*ratio+cp[3]*math.sqrt(ratio)
			if color_plan==1:
				color_ratio=col_ind%1
			else:
				col_ratio = 1-abs(col_ind%2-1)
			if color_plan=='1':
				self.setcolor(color_ratio,0,0,rainbow=True)
			elif color_plan=='2':
				self.setcolor(col_ratio*255,col_ratio*255,col_ratio*255)
			elif color_plan=='3':
				self.setcolor(random.randint(0,255),random.randint(0,255),random.randint(0,255))
			elif color_plan=='4':
				self.setcolor(r1*(1-col_ratio)+r2*col_ratio,g1*(1-col_ratio)+g2*col_ratio,b1*(1-col_ratio)+b2*col_ratio)
			if curve_type=='0':
				self.polar(lambda x: 50*(x-2*math.pi*t)/abs(lpn), start=2*math.pi*t, loop=lpn, dir=1)
			elif curve_type=='1':
				self.polar(lambda x: 50*(x-2*math.pi*t)/abs(lpn), start=2*math.pi*t, loop=lpn, dir=-1)
			elif curve_type=='2':
				self.polar(lambda x: 300*(t+0.01), cx=a, cy=b)
			elif curve_type=='3':
				R = 280
				r = 0.6*R*t+10
				theta = math.asin(math.sqrt((R*R-r*r)/(R*R+r*r*t*t)))
				self.paraplot(lambda x: r/math.cos(x), lambda y: t*r*math.tan(y), start=-theta,end=theta)
				self.paraplot(lambda x: t*r*math.tan(x), lambda y: r/math.cos(y), start=-theta,end=theta)
				self.paraplot(lambda x: -r/math.cos(x), lambda y: t*r*math.tan(y), start=-theta,end=theta)
				self.paraplot(lambda x: t*r*math.tan(x), lambda y: -r/math.cos(y), start=-theta,end=theta)
			elif curve_type=='4':
				R = 280
				r = R*(1-t)**0.9
				theta = math.asin(math.sqrt((R*R-r*r)/(R*R+r*r*t*t)))
				self.paraplot(lambda x: r/math.cos(x), lambda y: t*r*math.tan(y), start=-theta,end=theta)
				self.paraplot(lambda x: t*r*math.tan(x), lambda y: r/math.cos(y), start=-theta,end=theta)
				self.paraplot(lambda x: -r/math.cos(x), lambda y: t*r*math.tan(y), start=-theta,end=theta)
				self.paraplot(lambda x: t*r*math.tan(x), lambda y: -r/math.cos(y), start=-theta,end=theta)
			elif curve_type=='5':
				R = 280
				r=R*(1-t)
				theta = math.asin(math.sqrt((R*R-r*r)/(R*R+r*r)))
				self.paraplot(lambda x: r/math.cos(x), lambda y: r*math.tan(y), start=-theta,end=theta)
				self.paraplot(lambda x: r*math.tan(x), lambda y: r/math.cos(y), start=-theta,end=theta)
				self.paraplot(lambda x: -r/math.cos(x), lambda y: r*math.tan(y), start=-theta,end=theta)
				self.paraplot(lambda x: r*math.tan(x), lambda y: -r/math.cos(y), start=-theta,end=theta)
			elif curve_type=='6':
				self.paraplot(lambda x: 100*(t+1)*((1-k)*math.cos(x)+l*k*math.cos((1-k)*x/k))/(1-k+l*k), 
				lambda y: 100*(t+1)*((1-k)*math.sin(y)-l*k*math.sin((1-k)*y/k))/(1-k+l*k), start=0, end=lpn*2*math.pi)
		self.mouse.move(a,b)

	def paraplot(self,fx,fy,start=0,end=2*math.pi,cx=0,cy=0,speed=1):
		if cx==0:
			cx=self.center_x
			cy=self.center_y
		t=start
		self.mouse.press(cx+round(fx(t)), cy+round(fy(t)))
		time.sleep(0.1)
		while t<=end:
			self.mouse.drag(cx+round(fx(t)), cy+round(fy(t)))
			if (end-start)<1 and (end-start)>0.4:
				dt=speed*(end-start)/50
			else:
				dt=speed*0.01
			i=2
			for i in range(2,10):
				dst = 0
				for j in range(1,i):
					r = 300/(math.sqrt(fx(t)*fx(t)+fy(t)*fy(t))+200)
					x1 = fx(t)
					y1 = fy(t)
					x2 = fx(t+i*dt)
					y2 = fy(t+i*dt)
					x3 = fx(t+j*dt)
					y3 = fy(t+j*dt)
					if x1==x2 and y1==y2:
						dst=0
					else:
						dst = abs(x3*(y1-y2)-y3*(x1-x2)+x1*y2-x2*y1)/math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
					if dst > 0.05/r:
						break
				if dst>0.05/r:
					break
			t+= (i-1)*dt
			time.sleep(0.02)
		self.mouse.drag(cx+round(fx(end)),cy+round(fy(end)))
		time.sleep(0.1)
		self.mouse.release(cx+round(fx(end)),cy+round(fy(end)))

	def polar(self,f,start=0,loop=1,speed=1,cx=0,cy=0,dir=1):
		theta=start
		if loop<0:
			end=-2*math.pi*loop
		else:
			end=start+loop*2*math.pi
		if cx==0:
			cx=self.center_x
			cy=self.center_y
		self.mouse.press(round(cx + f(start)*math.cos(start)), round(cy-dir*f(start)*math.sin(start)))
		mindst = 2
		while theta<=end:
			x = round(cx + f(theta)*math.cos(theta))
			y = round(cy - dir*f(theta)*math.sin(theta))
			self.mouse.drag(x,y)
			dtheta=0.009*speed
			i=2
			for i in range(2,10):
				dst = 0
				for j in range(1,i):
					x1 = f(theta)*math.cos(theta)
					y1 = f(theta)*math.sin(theta)
					x2 = f(theta+i*dtheta)*math.cos(theta+i*dtheta)
					y2 = f(theta+i*dtheta)*math.sin(theta+i*dtheta)
					x3 = f(theta+j*dtheta)*math.cos(theta+j*dtheta)
					y3 = f(theta+j*dtheta)*math.sin(theta+j*dtheta)
					if x1==x2 and y1==y2:
						dst=1
					else:
						dst = abs(x3*(y1-y2)-y3*(x1-x2)+x1*y2-x2*y1)/math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
					if dst > 0.05:
						break
				if dst>0.05:
					break
			theta += (i-1)*dtheta
			time.sleep(0.01)
		x=cx+round(f(end)*math.cos(end))
		y=cy-dir*round(f(end)*math.sin(end))
		self.mouse.drag(x,y)
		time.sleep(0.1)
		self.mouse.release(x,y)

	def autoplot(self,mode=1):
		a,b = self.center_x, self.center_y
		target = [(250,15,225)]
		if mode>0:
			self.mouse.move(1,100)
			time.sleep(2)
			im=grab()
			bad_count=0
			pix=[[0 for i in range(14)]for j in range(14)]
			tot=0
			for j in range(210):
				for i in range(210):
					x=a-210+i*2
					y=b-210+j*2
					r0,g0,b0 = im.getpixel((x,y))
					for colors in target:
						r1,g1,b1=colors
						if max(abs(r0-r1),abs(g0-g1),abs(b0-b1))<40:
							pix[j//15][i//15]=1
							tot+=1
							break
			if tot<4000:
				return 0
			find_square=0
			for size in range(13,4,-1):
				for y in range(14-size):
					for x in range(14-size):
						gridcount=sum([sum(pix[y+k][x:x+size]) for k in range(size)])
						if gridcount==size*size:
							find_square=1
							cx = a-210+15*(2*x+size+1)
							cy = b-210+15*(2*y+size+1)
							radius = 15*size - 15
							break
					if find_square==1:
						break
				if find_square==1:
					break
			if find_square==0:
				return 0
		n=random.randint(4,8)
		theta=random.randint(0,360)
		if mode<0:
			cx,cy,radius=a,b,-mode
		if mode==0:
			self.setcolor(random.randint(0,255),random.randint(0,255),random.randint(0,255))
		for i in range(2*n+1):
			angle=math.pi*theta/180
			x0=cx+int(radius*math.cos(angle))
			y0=cy+int(radius*math.sin(angle))
			theta+=360*n/(2*n+1)
			angle=math.pi*theta/180
			x1=cx+int(radius*math.cos(angle))
			y1=cy+int(radius*math.sin(angle))
			self.drawline(x0,y0,x1,y1)
		return 1
						
	def MeteorShower(self,num=30):
		h_c=0
		v_c=0
		print('Please enable full screen in your brower and zoom in to an area contaminated by pink')
		print('Press enter when you are ready.')
		c=input()
		if c!='m':
			self.setcolor(255,255,255)
		if c=='q':
			while True:
				c=input()
				if c=='e':
					break
				try:
					n=int(c)
				except:
					n=200
				pen.autoplot(mode=-n)
		bad=0
		while True:
			h_c+=1
			a=pen.autoplot()
			if a==0:
				bad+=1
			else:
				bad=0
			time.sleep(1)
			if h_c == num or bad==5:
				bad=-1
				h_c=0
				v_c+=1
				pen.shift('down',1.2)
			elif v_c%2 == 0:
				pen.shift('right',1)
			else:
				pen.shift('left',1)
			time.sleep(1)

#main
m=PyMouse()
pen = paint(m)
while True:
	system('cls')
	print('*'*46)
	print('*  PigScript Special Build -- Meteor Shower  *')
	print('*'*46)
	print(' 1. Draw a local picture\n 2. Draw a website picture\n 3. Draw curves\n 4. Meteor Shower \n 5. Exit')
	while True:
		c=getch().decode()
		if c in '12345':
			source = int(c)
			break
	if source==5:
		system('cls')
		break
	elif source==4:
		pen.MeteorShower()
	elif source==3:
		pen.plotcurve()
		continue
	elif source==2:
		filename = input('URL of the picture: ')
	else:
		filename = input('Picture name: ')
	print('Enter your pen size (1-9): ')
	while True:
		c=getch().decode()
		if c in '0123456789':
			break
	sc = int(c)
	print('Please manually set your pen size to', c)
	c=getch().decode()
	tm = time.clock()
	pen.autodraw(filename, source, sc)
	sec = int(time.clock()-tm)
	print('Time elapsed {} minutes {} seconds'.format(sec//60, sec%60))
	input()
