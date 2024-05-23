# import noise
import numpy as np
from matplotlib import pyplot as plt
import random
import time
from perlin_noise import PerlinNoise
# import perlin


shape = (100, 100)
scale = 100
octaves = 1
persistence = 1
lacunarity = 1
seed = np.random.randint(0,100)


def perlin(shape, scale, octaves, persistence, lacunarity, seed):
    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.pnoise2(i / scale,
                                        j / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=1024,
                                        repeaty=1024,
                                        base=seed)
    return world


plt.imshow(perlin(shape, scale, octaves, persistence, lacunarity, seed), cmap='gray')
plt.show()



noise1 = PerlinNoise(octaves=0.75)
noise2 = PerlinNoise(octaves=1)
noise3 = PerlinNoise(octaves=2)
noise4 = PerlinNoise(octaves=8)

def genPerlin(xp, yp, strength, noise):
    pic = []
    for i in range(xpix):
        row = []
        for j in range(ypix):
            noise_val = strength * noise([i / xp, j / yp])
            row.append(noise_val)
        pic.append(row)
    return(pic)

pic = genPerlin(50,50, 1, noise2)


"""
xpix, ypix = 80, 80
pic = []


def simplePerlin1(xpix, ypix):
    pic = []
    noise1 = PerlinNoise(octaves=1)
    noise2 = PerlinNoise(octaves=4)
    noise3 = PerlinNoise(octaves=8)
    noise4 = PerlinNoise(octaves=16)
    for i in range(xpix):
        row = []
        for j in range(ypix):
            noise_val = 3 * noise1([i / xpix, j / ypix])
            noise_val += 0.5 * noise2([i / xpix, j / ypix])
            noise_val += 0.25 * noise3([i / xpix, j / ypix])
            noise_val += 0.05 * noise4([i / xpix, j / ypix])
            row.append(noise_val)
        pic.append(row)
    return pic


def simplePerlin2(xpix, ypix):
    pic = []
    noise1 = PerlinNoise(octaves=1)
    noise2 = PerlinNoise(octaves=4)

    for i in range(xpix):
        row = []
        for j in range(ypix):
            noise_val = noise1([i / xpix, j / ypix])
            noise_val += 0.5 * noise2([i / xpix, j / ypix])
            noise_val += noise1([i / xpix, j / ypix])
            noise_val += noise1([i / xpix, j / ypix])
            noise_val += noise1([i / xpix, j / ypix])
            noise_val += noise1([i / xpix, j / ypix])
            noise_val += 0.5 * noise2([i / xpix, j / ypix])
            noise_val += 0.5 * noise2([i / xpix, j / ypix])
            noise_val += 0.5 * noise2([i / xpix, j / ypix])
            row.append(noise_val)
        pic.append(row)
    return pic


elevation = simplePerlin1(xpix + 25, ypix + 25)
hardness = simplePerlin2(xpix, ypix)
genriver = np.zeros((xpix + 25, ypix + 25))


def take_first(elem):
    try:
        return elem[0]
    except:
        return elem


class River:
    def __init__(self, elevation, hardness, startpos, depth, riverg, children=[]):
        self.elevation = elevation
        self.hardness = hardness
        self.riverg = riverg
        self.x = startpos[0]
        self.y = startpos[1]
        self.depth = depth
        self.children = children

    def flow(self):
        self.depth = self.depth - 1
        brk = False
        while brk == False:
            dirs = [0, 0, 0, 0]
            try:
                # Check if the river goes downhill
                n = [(self.elevation[self.y + 1][self.x]), [self.y + 1, self.x]]
            except:
                # Make sure the river doesent flow there
                n = self.elevation[self.y][self.x] + 10
            try:
                # Check if the river goes downhill
                s = [(self.elevation[self.y - 1][self.x]), [self.y - 1, self.x]]
            except:
                # Make sure the river doesent flow there
                s = self.elevation[self.y][self.x] + 10
            try:
                # Check if the river goes downhill
                e = [(self.elevation[self.y][self.x + 1]), [self.y, self.x + 1]]
            except:
                # Make sure the river doesent flow there
                e = self.elevation[self.y][self.x] + 10
            try:
                # Check if the river goes downhill
                w = [(self.elevation[self.y][self.x - 1]), [self.y, self.x - 1]]
            except:
                # Make sure the river doesent flow there
                w = (self.elevation[self.y][self.x]) + 10

            # Check where the river could go, creating list of directions

            # def take_first(elem):
            #    return elem[0]

            # pick several river paths
            # dirs = [n,s,e,w]
            # dirs = sorted(dirs, key=take_first)
            # #print(dirs)
            # next = []
            # #print(self.elevation[self.y][self.x])
            # for yc in range(0, len(dirs)-1):
            #     #print(yc)
            #     if (dirs[yc][0]) >= (self.elevation[self.y][self.x]):
            #         del dirs[yc]
            #     else:
            #         next.append(dirs[yc][1])
            #
            # for elem in next:
            #     #print(self.river)
            #     self.riverg[elem[0]][elem[1]] = 1
            # Pick 1 path
            dirs = [n, s, e, w]
            dirs = sorted(dirs, key=take_first)
            next = []
            yc = 3
            dirlen = len(dirs)
            # Is the highest point high enough?
            if dirs[0][0] <= self.elevation[self.y][self.x]:
                # Send next coords
                next = (dirs[0][1])
            else:
                self.riverg[self.y][self.x] = 0.5
            for x in range(0, len(next)):
                # if not(self.riverg[next[0]][next[1]] == 1):
                self.riverg[next[0]][next[1]] = 1
                self.y = next[0]
                self.x = next[1]
                brk = True
            else:
                # todo Lakes
                pass
            brk = True
        # We've checked where the river can flow, time to make it flow!
        # a 2d list showing the rivers
        self.riverg[self.y][self.x] = 1

        if self.depth > 0:
            # for elem2 in next:
            if True:
                return self.flow()
        else:
            return self.riverg
    def setrivgraph(self, rivg):
        self.riverg = rivg
    def spawnrivers(self, repeat):
        r = river(self.elevation, hardness, [random.randint(20, 80), random.randint(20, 80)], 20, genriver)
        rivgraph = r.flow()
        while isinstance(rivgraph, river) == True:
            rivgraph = rivgraph.flow()
        rivs = []
        for x in repeat:
            rivs.append(river(self.elevation, hardness, [random.randint(20, 80), random.randint(20, 80)], 10, self.riverg).flow())
    def erodegraph(self, strength):
        for y in range(0, len(elevation)):
            for x in range(0, len(elevation[y])):
                if self.riverg[y, x] == 1:
                    # elevation[y][x] -= 0.0005
                    self.elevation[y][x] -= 0.0000
                    # Erode surroundings
                    self.elevation[y + 1][x] -= strength
                    self.elevation[y - 1][x] -= strength
                    self.elevation[y][x + 1] -= strength
                    self.elevation[y][x - 1] -= strength
        return [self.riverg, self.elevation]
"""

freq = 25
for x in range(0, freq):
    xpos = random.randint(0, xpix)
    ypos = random.randint(0, ypix)
# r = river(elevation, hardness, [random.randint(0,xpix),random.randint(0,ypix)], 15, genriver)
"""
def runriver(rivgraph, elevation, strength):
    r = river(elevation, hardness, [random.randint(20, 80), random.randint(20, 80)], 20, genriver)
    rivgraph = r.flow()
    while isinstance(rivgraph, river) == True:
        rivgraph = rivgraph.flow()
    rivs = []
    rivs.append(river(elevation, hardness, [random.randint(20, 80), random.randint(20, 80)], 10, rivgraph).flow())
    rivs.append(river(elevation, hardness, [random.randint(20, 80), random.randint(20, 80)], 10, rivs[0]).flow())
    rivs.append(river(elevation, hardness, [random.randint(20, 80), random.randint(20, 80)], 10, rivs[0]).flow())
    rivs.append(river(elevation, hardness, [random.randint(20, 80), random.randint(20, 80)], 10, rivs[0]).flow())
    rivs.append(river(elevation, hardness, [random.randint(20, 80), random.randint(20, 80)], 10, rivs[0]).flow())
    rivs.append(river(elevation, hardness, [random.randint(20, 80), random.randint(20, 80)], 10, rivs[0]).flow())
    for y in range(0, len(elevation)):
        for x in range(0, len(elevation[y])):
            if rivgraph[y, x] == 1:
                # elevation[y][x] -= 0.0005
                elevation[y][x] -= 0.0000
                # Erode surroundings
                elevation[y + 1][x] -= strength
                elevation[y - 1][x] -= strength
                elevation[y][x + 1] -= strength
                elevation[y][x - 1] -= strength
    return [rivgraph, elevation]
ret = runriver(genriver, elevation, 0.00025)
"""
"""
rivgraph = ret[0]
elevation = ret[1]
# Create lots of rivers
for x in range(0, 100):
    ret = runriver(rivgraph, elevation)
    # rivgraph = ret[0]
    # elevation = ret[1]
"""
for x in range(0,1):
    rivgraph = ret[0]
    elevation = ret[1]
    for x in range(0, 100):
        ret = runriver(rivgraph, elevation, 0.00025)
rivgraph = ret[0]
elevation = ret[1]
"""
# Reset river
def reset_to_0(the_array):
    for i, e in enumerate(the_array):
        if isinstance(e, list):
            reset_to_0(e)
        else:
            the_array[i] = 0
reset_to_0(rivgraph)
for x in range(0, 1):
    ret = runriver(rivgraph, elevation, 0.00001)
rivgraph = ret[0]
elevation = ret[1]
"""
f, ax = plt.subplots(1, 2)
ax[0].imshow(rivgraph, cmap="gray")  # first image
ax[1].imshow(elevation)  # second image
plt.show()