# Meteor Shower Data Plotter 
Hi, human beings! It's December again, which means we can enjoy one of the biggest meteor showers very soon! However, the sad thing is that the moon will be around Gemini during the show, which will be unfortunately causing possible blindness during stargazing.

I am a college student interested in python and astronomy, so I decided to write a short program to graph the average number of visual meteors on an hourly basis.

The data I am using is from International Meteor Organization. You can download any visual meteor database you like.
The data is free but you need to register as a free member first.
The link: https://www.imo.net/members/imo_vmdb/download

I put the data files in one directory named MeteorShower and the program will read through the directory. I am using the data of Perseids from last 30 years. You can try to download other meteor shower databases such as the Perseids.

At the beginning of the program, you should choose the meteorshower name and the month you want to observe.
For example: 

	name_of_shower = 'PER'  (PER for Perseids, names are shown on the file name)
	month_of_observation = '08' (Please fill in two digits)
	DATA_PATH = r'YOUR FILE DIRECTORY'

Let's enjoy the meteorshowers!
