#include <cmath>
#include <cstdlib>
#include "perlin.hpp"

const int table[] = {151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180};


Perlin::Perlin()
{
  m_seed = rand();
  for (int i=0;i<256;i++) {
    m_permTable[i] = i;
  }
  for (int i=0;i<256;i++) {
    int random = rand()&255;
    int temp = m_permTable[i];
    m_permTable[i] = m_permTable[random];
    m_permTable[random] = temp;
    // m_permTable[i] = table[i];
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
  return {(float)cos(g), (float)sin(g)};
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
