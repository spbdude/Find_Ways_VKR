import cv2

import numpy as np
import pickle
import pyastar2d
import networkx as nx


def dist(a, b):
  (x1, y1) = a
  (x2, y2) = b
  return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def FindWayPoints2(num, ways2, ways3, flag):
  h = imge(num)
  check = {'1.0': 0,
           '1.1': 1,
           '2.5': 2,
           '4.0': 3,
           '8.0': 4,
           '9.0': 5}
  wayout = []
  i = 0
  p = list(num)
  a, b = int(p[0]), int(p[1])
  if a == 0 or flag == 1:
    while i != (512):
      wayout1 = []
      wayout2 = []
      wayout3 = []
      wayout4 = []
      wayout5 = []
      wayout6 = []
      check2 = [0, 0, 0, 0, 0, 0]
      while h[0][i] != np.inf:
        wayoutloc = i
        # print(i)
        if i != 511:
          while h[0][i] == h[0][i + 1]:
            if i == 510:
              break
            i += 1
        l = (i + wayoutloc) // 2
        o = check.get(str(h[0][l]))
        check2[o] = 1
        if i != 511 and wayoutloc != 0:
          if h[0][i] < h[0][i + 1] and h[0][i] < h[0][wayoutloc - 1]:
            if o == 0:
              wayout1.append((0, l))
            elif o == 1:
              wayout2.append((0, l))
            elif o == 2:
              wayout3.append((0, l))
            elif o == 3:
              wayout4.append((0, l))
            elif o == 4:
              wayout5.append((0, l))
        elif i == 511:
          if h[0][i] < h[0][wayoutloc - 1]:
            if o == 0:
              wayout1.append((0, l))
            elif o == 1:
              wayout2.append((0, l))
            elif o == 2:
              wayout3.append((0, l))
            elif o == 3:
              wayout4.append((0, l))
            elif o == 4:
              wayout5.append((0, l))
        elif wayoutloc == 0:
          if h[0][i] < h[0][i + 1]:
            if o == 0:
              wayout1.append((0, l))
            elif o == 1:
              wayout2.append((0, l))
            elif o == 2:
              wayout3.append((0, l))
            elif o == 3:
              wayout4.append((0, l))
            elif o == 4:
              wayout5.append((0, l))
        i += 1
        if i == 511:
          break
      anotherDic = {0: wayout1, 1: wayout2, 2: wayout3, 3: wayout4, 4: wayout5, 5: wayout6}
      for r in range(len(check2) - 1):
        if check2[r] != 0:
          first = anotherDic.get(r)
          u = r + 1
          second = anotherDic.get(u)
          wayout.extend(first)
          wayout.extend(second)
          break
      i += 1
  else:
    a1 = a - 1
    num1 = str(a1) + str(b)
    k = ways3.get(num1)
    wayout.extend(k)
  i = 0
  wayoutput2 = []
  while i != (512):
    wayout1 = []
    wayout2 = []
    wayout3 = []
    wayout4 = []
    wayout5 = []
    wayout6 = []
    wayoutput21 = []
    wayoutput22 = []
    wayoutput23 = []
    wayoutput24 = []
    wayoutput25 = []
    wayoutput26 = []
    check2 = [0, 0, 0, 0, 0, 0]
    while h[i][511] != np.inf:
      wayoutloc = i
      if wayoutloc != 511:
        while h[i][511] == h[i + 1][511]:
          if i == 510:
            break
          i += 1
      l = (i + wayoutloc) // 2
      o = check.get(str(h[l][511]))
      check2[o] = 1
      if i != 511 and wayoutloc != 0:
        if h[i][511] < h[i + 1][511] and h[i][511] < h[wayoutloc - 1][511]:
          if o == 0:
            wayout1.append((l, 511))
            wayoutput21.append((l, 0))
          elif o == 1:
            wayout2.append((l, 511))
            wayoutput22.append((l, 0))
          elif o == 2:
            wayout3.append((l, 511))
            wayoutput23.append((l, 0))
          elif o == 3:
            wayout4.append((l, 511))
            wayoutput24.append((l, 0))
          elif o == 4:
            wayout5.append((l, 511))
            wayoutput25.append((l, 0))
      elif i == 511:
        if h[i][511] < h[wayoutloc - 1][511]:
          if o == 0:
            wayout1.append((l, 511))
            wayoutput21.append((l, 0))
          elif o == 1:
            wayout2.append((l, 511))
            wayoutput22.append((l, 0))
          elif o == 2:
            wayout3.append((l, 511))
            wayoutput23.append((l, 0))
          elif o == 3:
            wayout4.append((l, 511))
            wayoutput24.append((l, 0))
          elif o == 4:
            wayout5.append((l, 511))
            wayoutput25.append((l, 0))
      elif wayoutloc == 0:
        if h[i][511] < h[i + 1][511]:
          if o == 0:
            wayout1.append((l, 511))
            wayoutput21.append((l, 0))
          elif o == 1:
            wayout2.append((l, 511))
            wayoutput22.append((l, 0))
          elif o == 2:
            wayout3.append((l, 511))
            wayoutput23.append((l, 0))
          elif o == 3:
            wayout4.append((l, 511))
            wayoutput24.append((l, 0))
          elif o == 4:
            wayout5.append((l, 511))
            wayoutput25.append((l, 0))
      i += 1
      if i == 511:
        break
    anotherDic = {0: wayout1, 1: wayout2, 2: wayout3, 3: wayout4, 4: wayout5, 5: wayout6}
    anotherDic2 = {0: wayoutput21, 1: wayoutput22, 2: wayoutput23, 3: wayoutput24, 4: wayoutput25, 5: wayoutput26}
    for o in range(len(check2) - 1):
      if check2[o] != 0:
        first = anotherDic.get(o)
        u = o + 1
        second = anotherDic.get(u)
        first2 = anotherDic2.get(o)
        second2 = anotherDic2.get(u)
        wayout.extend(first)
        wayout.extend(second)
        wayoutput2.extend(first2)
        wayoutput2.extend(second2)
        break
    i += 1
  i = 0
  ways2[num] = wayoutput2
  wayoutput3 = []
  while i != (512):
    wayout1 = []
    wayout2 = []
    wayout3 = []
    wayout4 = []
    wayout5 = []
    wayout6 = []
    wayoutput21 = []
    wayoutput22 = []
    wayoutput23 = []
    wayoutput24 = []
    wayoutput25 = []
    wayoutput26 = []
    check2 = [0, 0, 0, 0, 0, 0]
    while h[511][i] != np.inf:
      wayoutloc = i
      if wayoutloc != 511:
        while h[511][i] == h[511][i + 1]:
          if i == 510:
            break
          i += 1
      l = (i + wayoutloc) // 2
      o = check.get(str(h[511][l]))
      check2[o] = 1
      if i != 511 and wayoutloc != 0:
        if h[511][i] < h[511][i + 1] and h[511][i] < h[511][wayoutloc - 1]:
          if o == 0:
            wayout1.append((511, l))
            wayoutput21.append((0, l))
          elif o == 1:
            wayout2.append((511, l))
            wayoutput22.append((0, l))
          elif o == 2:
            wayout3.append((511, l))
            wayoutput23.append((0, l))
          elif o == 3:
            wayout4.append((511, l))
            wayoutput24.append((0, l))
          elif o == 4:
            wayout5.append((511, l))
            wayoutput25.append((0, l))
      elif i == 511:
        if h[511][i] < h[511][wayoutloc - 1]:
          if o == 0:
            wayout1.append((511, l))
            wayoutput21.append((0, l))
          elif o == 1:
            wayout2.append((511, l))
            wayoutput22.append((0, l))
          elif o == 2:
            wayout3.append((511, l))
            wayoutput23.append((0, l))
          elif o == 3:
            wayout4.append((511, l))
            wayoutput24.append((0, l))
          elif o == 4:
            wayout5.append((511, l))
            wayoutput25.append((0, l))
      elif wayoutloc == 0:
        if h[511][i] < h[511][i + 1]:
          if o == 0:
            wayout1.append((511, l))
            wayoutput21.append((0, l))
          elif o == 1:
            wayout2.append((511, l))
            wayoutput22.append((0, l))
          elif o == 2:
            wayout3.append((511, l))
            wayoutput23.append((0, l))
          elif o == 3:
            wayout4.append((511, l))
            wayoutput24.append((0, l))
          elif o == 4:
            wayout5.append((511, l))
            wayoutput25.append((0, l))
      i += 1
      if i == 511:
        break
    anotherDic = {0: wayout1, 1: wayout2, 2: wayout3, 3: wayout4, 4: wayout5, 5: wayout6}
    anotherDic2 = {0: wayoutput21, 1: wayoutput22, 2: wayoutput23, 3: wayoutput24, 4: wayoutput25, 5: wayoutput26}
    for o in range(len(check2) - 1):
      if check2[o] != 0:
        first = anotherDic.get(o)
        u = o + 1
        second = anotherDic.get(u)
        first2 = anotherDic2.get(o)
        second2 = anotherDic2.get(u)
        wayout.extend(first)
        wayout.extend(second)
        wayoutput3.extend(first2)
        wayoutput3.extend(second2)
        break
    i += 1
  ways3[num] = wayoutput3
  i = 0
  if b == 0 or flag == 1:
    while i != (512):
      wayout1 = []
      wayout2 = []
      wayout3 = []
      wayout4 = []
      wayout5 = []
      wayout6 = []
      check2 = [0, 0, 0, 0, 0, 0]
      while h[i][0] != np.inf:
        wayoutloc = i
        if wayoutloc != 511:
          while h[i][0] == h[i + 1][0]:
            if i == 510:
              break
            i += 1
        l = (i + wayoutloc) // 2
        o = check.get(str(h[l][0]))
        check2[o] = 1
        if i != 511 and wayoutloc != 0:
          if h[i][0] < h[i + 1][0] and h[i][0] < h[wayoutloc - 1][0]:
            if o == 0:
              wayout1.append((l, 0))
            elif o == 1:
              wayout2.append((l, 0))
            elif o == 2:
              wayout3.append((l, 0))
            elif o == 3:
              wayout4.append((l, 0))
            elif o == 4:
              wayout5.append((l, 0))
        elif i == 511:
          if h[i][0] < h[wayoutloc - 1][0]:
            if o == 0:
              wayout1.append((l, 0))
            elif o == 1:
              wayout2.append((l, 0))
            elif o == 2:
              wayout3.append((l, 0))
            elif o == 3:
              wayout4.append((l, 0))
            elif o == 4:
              wayout5.append((l, 0))
        elif wayoutloc == 0:
          if h[i][0] < h[i + 1][0]:
            if o == 0:
              wayout1.append((l, 0))
            elif o == 1:
              wayout2.append((l, 0))
            elif o == 2:
              wayout3.append((l, 0))
            elif o == 3:
              wayout4.append((l, 0))
            elif o == 4:
              wayout5.append((l, 0))
        i += 1
        if i == 511:
          break
      anotherDic = {0: wayout1, 1: wayout2, 2: wayout3, 3: wayout4, 4: wayout5, 5: wayout6}
      for r in range(len(check2) - 1):
        if check2[r] != 0:
          first = anotherDic.get(r)
          u = r + 1
          second = anotherDic.get(u)
          wayout.extend(first)
          wayout.extend(second)
          break
      i += 1
  else:
    b1 = b - 1
    num1 = str(a) + str(b1)
    k = ways2.get(num1)
    wayout.extend(k)
  if not wayout:
    flag = 2
  return wayout, ways2, ways3, flag

def imge(num):
  with open(num + '.npy', 'rb') as f:
    g = np.load(f)
  return(g)

def DrawWay(img, path):
    i = 0
    while i < len(path):
        x, y = path[i]
        img[x, y] = (255, 255, 255)
        img[x-1, y] = (255, 255, 255)
        i += 1
    return img

def procimg(num, img):
  only_red_pixels = np.argwhere(cv2.inRange(img, (0, 19, 61), (255,32, 255)))
  red_pixels = np.argwhere(cv2.inRange(img, (0, 19, 61), (255, 58, 255)))
  orange_pixels = np.argwhere(cv2.inRange(img, (0, 19, 61), (255, 163, 255)))
  yellow_pixels = np.argwhere(cv2.inRange(img, (0, 19, 61), (255, 255, 255)))
  blue_pixels = np.argwhere(cv2.inRange(img, (0, 82, 0), (255, 255, 255)))
  purple_pixels = np.argwhere(cv2.inRange(img, (0, 16, 0), (255, 255, 255)))
  # mass = np.zeros((512, 512), dtype=np.float32)
  mass = np.full((512, 512), np.inf, dtype=np.float32)
  for px, py in purple_pixels:
    mass[px][py] = 9
  for px, py in blue_pixels:
    mass[px][py] = 8
  for px, py in yellow_pixels:
    mass[px][py] = 4
  for px, py in orange_pixels:
    mass[px][py] = 2.5
  for px, py in red_pixels:
    mass[px][py] = 1.1
  for px, py in only_red_pixels:
    mass[px][py] = 1
  # print(mass[30][0])
  # print(list(mass))
  # mass = mass.astype(int)
  # with open('mass.txt', 'w') as fp:
  #   fp.write(str(mass))
  with open(num + '.npy', 'wb') as f:
    np.save(f, mass)
  return(mass)


points = {}
paths = {}
ways2 = {}
ways3 = {}
G = nx.Graph()

flag = 0
print('preprocessing')
for i in range(2):
  for j in range(4):
    num = str(i) + str(j)
    num2 = num +'.png'
    img = cv2.imread(num2, cv2.IMREAD_COLOR)
    try:
      procimg(num, img)
    except Exception:
      flag = 1
      continue
    points1, ways2, ways3, flag = FindWayPoints2(num, ways2, ways3, flag)
    points[num] = points1
    if flag == 2:
        continue
    flag = 0
    # print(points1)
    lenght = len(points1)
    graph1 = imge(num)
    # print('procimg + FW done!#################' + num)
    for w in range(lenght - 1):
      for t in range(w, lenght - 1):
        path = pyastar2d.astar_path(graph1, points1[w], points1[t+1], allow_diagonal=True)
        cost = 0
        for l in range(len(path)):
            px, py = path[l]
            cost += graph1[px][py]
        img = DrawWay(img, path)
        py1, px1 = points1[w]
        py2, px2 = points1[t + 1]
        if px1 == 0 and py1 == 0 and i != 0 and j != 0:
          g.add((py1 + 512 * i, px1 + 512 * j), (py1 + 512 * i, px1 + 512 * j - 1), 1.0, [(py1 + 512 * i, px1 + 512 * j), (py1 + 512 * i, px1 + 512 * j - 1)])
          g.add((py1 + 512 * i, px1 + 512 * j), (py1 + 512 * i - 1, px1 + 512 * j), 1.0, [(py1 + 512 * i, px1 + 512 * j), (py1 + 512 * i - 1, px1 + 512 * j)])
        elif px1 == 0 and j != 0 and py1 !=0:
          G.add_node((py1 + 512 * i, px1 + 512 * j))
          G.add_node((py1 + 512 * i, px1 + 512 * j - 1))
          G.add_edge((py1 + 512 * i, px1 + 512 * j), (py1 + 512 * i, px1 + 512 * j - 1), weight=1)
          if w == lenght - 2:
              G.add_node((py2 + 512 * i, px2 + 512 * j))
              G.add_node((py2 + 512 * i, px2 + 512 * j - 1))
              G.add_edge((py2 + 512 * i, px2 + 512 * j), (py2 + 512 * i, px2 + 512 * j - 1), weight=1)
        if py1 == 0 and i != 0 and px1 !=0:
          G.add_node((py1 + 512 * i, px1 + 512 * j))
          G.add_node((py1 + 512 * i - 1, px1 + 512 * j))
          G.add_edge((py1 + 512 * i, px1 + 512 * j), (py1 + 512 * i - 1, px1 + 512 * j), weight=1)
          if w == lenght - 2:
              G.add_node((py2 + 512 * i, px2 + 512 * j))
              G.add_node((py2 + 512 * i - 1, px2 + 512 * j))
              G.add_edge((py2 + 512 * i, px2 + 512 * j), (py2 + 512 * i - 1, px2 + 512 * j - 1), weight=1)
        G.add_node((py1 + 512 * i, px1 + 512 * j))
        G.add_node((py2 + 512 * i, px2 + 512 * j))
        G.add_edge((py1 + 512 * i, px1 + 512 * j), (py2 + 512 * i, px2 + 512 * j), weight=cost)
        paths[((py1 + 512 * i, px1 + 512 * j),(py2 + 512 * i, px2 + 512 * j))] = path
        paths[((py2 + 512 * i, px2 + 512 * j),(py1 + 512 * i, px1 + 512 * j))] = path
# with open("paths.pkl", "rb") as fp:
#   paths = pickle.load(fp)
# with open("points.pkl", "rb") as fp:
#    points = pickle.load(fp)
# with open("biggraph.pkl", "rb") as fp:
#    G = pickle.load(fp)


x1 = 338
y1 = 489
start = (y1, x1)
num1 = '10'
graph1 = imge(num1)
point = points.get(num1)
p = list(num1)
a, b = int(p[0]), int(p[1])
for t in range(len(point)):
  path = pyastar2d.astar_path(graph1, start, point[t], allow_diagonal=True)
  cost = 0
  for l in range(len(path)):
      px, py = path[l]
      cost += graph1[px][py]
  py2, px2 = point[t]
  G.add_edge((y1 + 512 * a, x1 + 512 * b), (py2 + 512 * a, px2 + 512 * b), weight=cost)
  paths[((y1 + 512 * a, x1 + 512 * b), (py2 + 512 * a, px2 + 512 * b))] = path
  paths[((py2 + 512 * a, px2 + 512 * b), (y1 + 512 * a, x1 + 512 * b))] = path
start = (y1 + 512 * a, x1 + 512 * b)
print('start is done')


x2 = 447
y2 = 33
end = (y2, x2)
num2 = '01'
graph1 = imge(num2)
point = points.get(num2)
p = list(num2)
a, b = int(p[0]), int(p[1])
for t in range(len(point)):
  path = pyastar2d.astar_path(graph1, end, point[t], allow_diagonal=True)
  cost = 0
  for l in range(len(path)):
      px, py = path[l]
      cost += graph1[px][py]
  py2, px2 = point[t]
  G.add_edge((y2 + 512 * a, x2 + 512 * b), (py2 + 512 * a, px2 + 512 * b), weight=cost)
  paths[((y2 + 512 * a, x2 + 512 * b), (py2 + 512 * a, px2 + 512 * b))] = path
  paths[((py2 + 512 * a, px2 + 512 * b), (y2 + 512 * a, x2 + 512 * b))] = path
end = (y2 + 512 * a, x2 + 512 * b)
print('end is done')

way = nx.astar_path(G, start, end, heuristic=dist, weight = 'weight')
for t in range(0, len(way), 2):
  a, b = way[t]
  # print(a,b, '*******')
  num1 = a//512
  num2 = b//512
  num = str(num1) + str(num2) + '.png'
  img5 = cv2.imread(num, cv2.IMREAD_COLOR)
  # print(way[t], way[t + 1])
  # print(paths.get((way[t],way[t+1])))
  img5 = DrawWay(img5, paths.get((way[t],way[t+1])))
  if t == 0:
    cv2.circle(img5, (338,489), 7, (255, 0, 0), 2)
  if t == 4:
    cv2.circle(img5, (447,33), 7, (0, 255, 0), 2)
  cv2.imshow('img', img5)

  cv2.waitKey(0)
  h = str(t)
  cv2.imwrite(h + 'picture.png', img5)




