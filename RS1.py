# REF:
# https://towardsdatascience.com/programmatic-identification-of-support-resistance-trend-lines-with-python-d797a4a90530

import numpy as np
import pandas as pd
import yfinance as yf #pip install yfinance
tick = yf.Ticker('^GSPC')
hist = tick.history(period="max", rounding=True)
#hist = hist[:'2019-10-07']
hist = hist[-1000:]
h = hist.Close.tolist()

minimaIdxs = np.flatnonzero(hist.Close.rolling(window=3, min_periods=1, center=True).aggregate(lambda x: len(x) == 3 and x[0] > x[1] and x[2] > x[1])).tolist()
maximaIdxs = np.flatnonzero(hist.Close.rolling(window=3, min_periods=1, center=True).aggregate(lambda x: len(x) == 3 and x[0] < x[1] and x[2] < x[1])).tolist()

hs = hist.Close.loc[hist.Close.shift(-1) != hist.Close]
x = hs.rolling(window=3, center=True).aggregate(lambda x: x[0] > x[1] and x[2] > x[1])
minimaIdxs = [hist.index.get_loc(y) for y in x[x == 1].index]
x = hs.rolling(window=3, center=True).aggregate(lambda x: x[0] < x[1] and x[2] < x[1])
maximaIdxs = [hist.index.get_loc(y) for y in x[x == 1].index]

from findiff import FinDiff #pip install findiff
dx = 1 #1 day interval
d_dx = FinDiff(0, dx, 1)
d2_dx2 = FinDiff(0, dx, 2)
clarr = np.asarray(hist.Close)
mom = d_dx(clarr)
momacc = d2_dx2(clarr)

def get_extrema(isMin):
  return [x for x in range(len(mom))
    if (momacc[x] > 0 if isMin else momacc[x] < 0) and
      (mom[x] == 0 or #slope is 0
        (x != len(mom) - 1 and #check next day
          (mom[x] > 0 and mom[x+1] < 0 and
           h[x] >= h[x+1] or
           mom[x] < 0 and mom[x+1] > 0 and
           h[x] <= h[x+1]) or
         x != 0 and #check prior day
          (mom[x-1] > 0 and mom[x] < 0 and
           h[x-1] < h[x] or
           mom[x-1] < 0 and mom[x] > 0 and
           h[x-1] > h[x])))]
minimaIdxs, maximaIdxs = get_extrema(True), get_extrema(False)

import findiff
coeff = findiff.coefficients(deriv=1, acc=2)
#print(coeff)

#print(hist)

hist = hist[:'2021-06-14']
day = 990 # September 27, 2019 per example (-7 or 7th last point)
sum([coeff['center']['coefficients'][x] * hist.Close[day + coeff['center']['offsets'][x]] for x in range(len(coeff['center']['coefficients']))])

coeff=findiff.coefficients(deriv=2, acc=2)
#print(coeff)
sum([coeff['center']['coefficients'][x] *
     hist.Close[day + coeff['center']['offsets'][x]]
     for x in range(len(coeff['center']['coefficients']))])

def get_bestfit3(x0, y0, x1, y1, x2, y2):
    xbar, ybar = (x0 + x1 + x2) / 3, (y0 + y1 + y2) / 3
    xb0, yb0, xb1, yb1, xb2, yb2 =    x0-xbar, y0-ybar, x1-xbar, y1-ybar, x2-xbar, y2-ybar
    xs = xb0*xb0+xb1*xb1+xb2*xb2
    m = (xb0*yb0+xb1*yb1+xb2*yb2) / xs
    b = ybar - m * xbar
    ys0, ys1, ys2 =    (y0 - (m * x0 + b)),(y1 - (m * x1 + b)),(y2 - (m * x2 + b))
    ys = ys0*ys0+ys1*ys1+ys2*ys2
    ser = np.sqrt(ys / xs)
    return m, b, ys, ser, ser * np.sqrt((x0*x0+x1*x1+x2*x2)/3)

def get_bestfit(pts):
  xbar, ybar = [sum(x) / len (x) for x in zip(*pts)]
  def subcalc(x, y):
      tx, ty = x - xbar, y - ybar
      return tx * ty, tx * tx, x * x
  (xy, xs, xx) =    [sum(q) for q in zip(*[subcalc(x, y) for x, y in pts])]
  m = xy / xs
  b = ybar - m * xbar
  ys = sum([np.square(y - (m * x + b)) for x, y in pts])
  ser = np.sqrt(ys / ((len(pts) - 2) * xs))
  return m, b, ys, ser, ser * np.sqrt(xx / len(pts))


ymin, ymax = [h[x] for x in minimaIdxs], [h[x] for x in maximaIdxs]
zmin, zmne, _, _, _ = np.polyfit(minimaIdxs, ymin, 1, full=True)  #y=zmin[0]*x+zmin[1]
pmin = np.poly1d(zmin).c
zmax, zmxe, _, _, _ = np.polyfit(maximaIdxs, ymax, 1, full=True) #y=zmax[0]*x+zmax[1]
pmax = np.poly1d(zmax).c
print((pmin, pmax, zmne, zmxe))

p, r = np.polynomial.polynomial.Polynomial.fit(minimaIdxs, ymin, 1, full=True) #more numerically stable
pmin, zmne = list(reversed(p.convert().coef)), r[0]
p, r = np.polynomial.polynomial.Polynomial.fit(maximaIdxs, ymax, 1, full=True) #more numerically stable
pmax, zmxe = list(reversed(p.convert().coef)), r[0]

def get_trend(Idxs):
  trend = []
  for x in range(len(Idxs)):
    for y in range(x+1, len(Idxs)):
      for z in range(y+1, len(Idxs)):
        trend.append(([Idxs[x], Idxs[y], Idxs[z]],
          get_bestfit3(Idxs[x], h[Idxs[x]],
                       Idxs[y], h[Idxs[y]],
                       Idxs[z], h[Idxs[z]])))
  return list(filter(lambda val: val[1][3] <= fltpct, trend))
mintrend, maxtrend = get_trend(minimaIdxs), get_trend(maximaIdxs)

def get_trend_opt(Idxs):
  slopes, trend = [], []
  for x in range(len(Idxs)): #O(n^2*log n) algorithm
    slopes.append([])
    for y in range(x+1, len(Idxs)):
      slope = (h[Idxs[x]] - h[Idxs[y]]) / (Idxs[x] - Idxs[y])
      slopes[x].append((slope, y))
  for x in range(len(Idxs)):
    slopes[x].sort(key=lambda val: val[0])
    CurIdxs = [Idxs[x]]
    for y in range(0, len(slopes[x])):
      CurIdxs.append(Idxs[slopes[x][y][1]])
      if len(CurIdxs) < 3: continue
      res = get_bestfit([(p, h[p]) for p in CurIdxs])
      if res[3] <= fltpct:
        CurIdxs.sort()
        if len(CurIdxs) == 3:
          trend.append((CurIdxs, res))
          CurIdxs = list(CurIdxs)
        else: CurIdxs, trend[-1] = list(CurIdxs), (CurIdxs, res)
      else: CurIdxs = [CurIdxs[0], CurIdxs[-1]] #restart search
  return trend
mintrend, maxtrend =  get_trend_opt(minimaIdxs), get_trend_opt(maximaIdxs)

def make_image(Idxs):
  max_size = int(np.ceil(2/np.tan(np.pi / (360 * 5)))) #~1146
  m, tested_angles =    hist.Close.min(), np.linspace(-np.pi / 2, np.pi / 2, 360*5)
  height = int((hist.Close.max() - m + 0.01) * 100)
  mx = min(max_size, height)
  scl = 100.0 * mx / height
  image = np.zeros((mx, len(hist))) #in rows, columns or y, x
  for x in Idxs:
    image[int((h[x] - m) * scl), x] = 255
  return image, tested_angles, scl, m

def hough_points(pts, width, height, thetas):
  diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
  # Cache some resuable values
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)
  num_thetas = len(thetas)
  # Hough accumulator array of theta vs rho
  accumulator =np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
  # Vote in the hough accumulator
  for i in range(len(pts)):
    x, y = pts[i]
    for t_idx in range(num_thetas):
      # Calculate rho. diag_len is added for a positive index
      rho=int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + diag_len
      accumulator[rho, t_idx] += 1
  return accumulator, thetas, rhos

def find_line_pts(Idxs, x0, y0, x1, y1):
  s = (y0 - y1) / (x0 - x1)
  i, dnm = y0 - s * x0, np.sqrt(1 + s*s)
  dist = [(np.abs(i+s*x-h[x])/dnm, x) for x in Idxs]
  dist.sort(key=lambda val: val[0])
  pts, res = [], None
  for x in range(len(dist)):
    pts.append((dist[x][1], h[dist[x][1]]))
    if len(pts) < 3: continue
    r = get_bestfit(pts)
    if r[3] > fltpct:
      pts = pts[:-1]
      break
    res = r
  pts = [x for x, _ in pts]
  pts.sort()
  return pts, res

def houghpt(Idxs):
  max_size = int(np.ceil(2/np.tan(np.pi / (360 * 5)))) #~1146
  m, tested_angles =    hist.Close.min(), np.linspace(-np.pi / 2, np.pi / 2, 360*5)
  height = int((hist.Close.max() - m + 1) * 100)
  mx = min(max_size, height)
  scl = 100.0 * mx / height
  acc, theta, d = hough_points(
    [(x, int((h[x] - m) * scl)) for x in Idxs], mx, len(hist),
    np.linspace(-np.pi / 2, np.pi / 2, 360*5))
  origin, lines = np.array((0, len(hist))), []
  for x, y in np.argwhere(acc >= 3):
    dist, angle = d[x], theta[y]
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    y0, y1 = y0 / scl + m, y1 / scl + m
    pts, res = find_line_pts(Idxs, 0, y0, len(hist), y1)
    if len(pts) >= 3: lines.append((pts, res))
  return lines
mintrend, maxtrend = houghpt(minimaIdxs), houghpt(maximaIdxs)

def hough(Idxs): #pip install scikit-image
  image, tested_angles, scl, m = make_image(Idxs)
  from skimage.transform import hough_line, hough_line_peaks
  h, theta, d = hough_line(image, theta=tested_angles)
  origin, lines = np.array((0, image.shape[1])), []
  for pts, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=2)):
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    y0, y1 = y0 / scl + m, y1 / scl + m
    pts, res = find_line_pts(Idxs, 0, y0, image.shape[1], y1)
    if len(pts) >= 3: lines.append((pts, res))
  return lines
mintrend, maxtrend = hough(minimaIdxs), hough(maximaIdxs)

def prob_hough(Idxs): #pip install scikit-image
  image, tested_angles, scl, m = make_image(Idxs)
  from skimage.transform import probabilistic_hough_line
  lines = []
  for x in range(hough_prob_iter):
    lines.append(probabilistic_hough_line(image, threshold=2,
                 theta=tested_angles, line_length=0,
      line_gap=int(np.ceil(np.sqrt(
        np.square(image.shape[0]) + np.square(image.shape[1]))))))
  l = []
  for (x0, y0), (x1, y1) in lines:
    if x0 == x1: continue
    if x1 < x0: (x0, y0), (x1, y1) = (x1, y1), (x0, y0)
    y0, y1 = y0 / scl + m, y1 / scl + m
    pts, res = find_line_pts(Idxs, x0, y0, x1, y1)
    if len(pts) >= 3: l.append((pts, res))
  return l
mintrend, maxtrend = prob_hough(minimaIdxs), prob_hough(maximaIdxs)


def measure_area(trendline, isMin):
  base = trendline[0][0]
  m, b, ser =    trendline[1][0], trendline[1][1], h[base:trendline[0][-1]+1]
  return sum([max(0, (m * (x+base) + b) - y
               if isMin else y - (m * (x+base) + b))
              for x, y in enumerate(ser)]) / len(ser)
mintrend = [(pts, (res[0], res[1], res[2], res[3], res[4],
             measure_area((pts, res), True)))
            for pts, res in mintrend]
maxtrend = [(pts, (res[0], res[1], res[2], res[3], res[4],
             measure_area((pts, res), False)))
            for pts, res in maxtrend]
mintrend.sort(key=lambda val: val[1][5])
maxtrend.sort(key=lambda val: val[1][5])
print((mintrend[:5], maxtrend[:5]))

def merge_lines(Idxs, trend):
  for x in Idxs:
    l = []
    for i, (p, r) in enumerate(trend):
      if x in p: l.append((r[0], i))
    l.sort(key=lambda val: val[0])
    if len(l) > 1: CurIdxs = list(trend[l[0][1]][0])
    for (s, i) in l[1:]:
      CurIdxs += trend[i][0]
      CurIdxs = list(dict.fromkeys(CurIdxs))
      CurIdxs.sort()
      res = get_bestfit([(p, h[p]) for p in CurIdxs])
      if res[3] <= fltpct: trend[i-1], trend[i], CurIdxs =        ([], None), (CurIdxs, res), list(CurIdxs)
      else: CurIdxs = list(trend[i][0]) #restart search from here
  return list(filter(lambda val: val[0] != [], trend))
mintrend, maxtrend = merge_lines(minimaIdxs, mintrend),merge_lines(maximaIdxs, maxtrend)



