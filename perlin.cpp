#include <cmath>
#include <cstdlib>
#include "perlin.hpp"

Perlin::Perlin()
{
  for (int i=0;i<256;i++) {
    m_permTable[i] = i;
  }
  for (int i=0;i<256;i++) {
    int random = rand()&255;
    int temp = m_permTable[i];
    m_permTable[i] = m_permTable[random];
    m_permTable[random] = temp;
  }
}

Perlin::Perlin(int scale):
m_scale(scale)
{ Perlin(); }

double Perlin::getPerlin(double x, double y) const
{
  y /= m_scale;
  x /= m_scale;
  int x0(((int)x)&255), y0(((int)y)&255);
  double g[4];
  g[0] = hash(x0, y0);
  g[1] = hash(x0+1, y0);
  g[2] = hash(x0, y0+1);
  g[3] = hash(x0+1, y0+1);
  double r = (x-x0)*cos(g[0])+(y-y0)*sin(g[0]);
  double s = (x-x0-1)*cos(g[1])+(y-y0)*sin(g[1]);
  double t = (x-x0)*cos(g[2])+(y-y0-1)*sin(g[2]);
  double u = (x-x0-1)*cos(g[3])+(y-y0-1)*sin(g[3]);
  r += cubic(x-x0)*(s-r);
  t += cubic(x-x0)*(u-t);
  return r+cubic(y-y0)*(t-r);
}

sf::Vector2f Perlin::getVector(double x, double y) {
  y /= m_scale;
  x /= m_scale;
  int x0(((int)x)&255), y0(((int)y)&255);
  double g = hash(x0, y0);
  return {cos(g), sin(g)};
}


double const u(sqrt(2)/2.0);
double grads[18][3] = {{u, u, 0}, {-u, u, 0}, {u, -u, 0}, {-u, -u, 0},
                    {u, 0, u}, {-u, 0, u}, {u, 0, -u}, {-u, 0, -u},
                    {0, u, u}, {0, -u, u}, {0, u, -u}, {0, -u, -u},
                    {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}};

double Perlin::getPerlin(double x, double y, double z) const
{
  y /= m_scale;
  x /= m_scale;
  z /= m_scale;
  int x0((int)x), y0((int)y), z0((int)z);
  x -= x0, y -= y0, z -= z0;
  double* g[8];
  g[0] = grads[hash(x0, y0, z0)%18];
  g[1] = grads[hash(x0+1, y0, z0)%18];
  g[2] = grads[hash(x0, y0+1, z0)%18];
  g[3] = grads[hash(x0+1, y0+1, z0)%18];
  g[4] = grads[hash(x0, y0, z0+1)%18];
  g[5] = grads[hash(x0+1, y0, z0+1)%18];
  g[6] = grads[hash(x0, y0+1, z0+1)%18];
  g[7] = grads[hash(x0+1, y0+1, z0+1)%18];
  double p000 = x*g[0][0]+y*g[0][1]+z*g[0][2];
  double p100 = (x-1)*g[1][0]+y*g[1][1]+z*g[1][2];
  double p010 = x*g[2][0]+(y-1)*g[2][1]+z*g[2][2];
  double p110 = (x-1)*g[3][0]+(y-1)*g[3][1]+z*g[3][2];
  double p001 = x*g[4][0]+y*g[4][1]+(z-1)*g[4][2];
  double p101 = (x-1)*g[5][0]+y*g[5][1]+(z-1)*g[5][2];
  double p011 = x*g[6][0]+(y-1)*g[6][1]+(z-1)*g[6][2];
  double p111 = (x-1)*g[7][0]+(y-1)*g[7][1]+(z-1)*g[7][2];

  double p00 = p000+cubic(z)*(p001-p000);
  double p10 = p100+cubic(z)*(p101-p100);
  double p01 = p010+cubic(z)*(p011-p010);
  double p11 = p110+cubic(z)*(p111-p110);

  double p0 = p00+cubic(y)*(p01-p00);
  double p1 = p10+cubic(y)*(p11-p10);
  return p0+cubic(x)*(p1-p0);
}

int Perlin::hash(int x, int y, int z) const {
  return z>0?m_permTable[(x+m_permTable[(y+m_permTable[z&255])&255])&255]:m_permTable[(x&255+m_permTable[y&255])&255];
}


void Perlin::setScale(double scale) { m_scale = scale; }

double Perlin::getscale() const { return m_scale; }

int Perlin::getTableNumber(int id) { return m_permTable[id]; }

double cubic(double x) {
  return 3*x*x-2*x*x*x;
  // return x;
}
