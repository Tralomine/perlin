#pragma once
#include <SFML/System.hpp>

class Perlin {
  public:
    Perlin();
    Perlin(int scale);
    double getPerlin(double x, double y) const;
    double getPerlin(double x, double y, double z) const;
    void setScale(double scale);
    double getscale() const;
    int getTableNumber(int id);
    sf::Vector2f getVector(double x, double y);
  private:
    int m_permTable[256];
    double m_scale;
    size_t m_seed;
    inline int hash(int x, int y, int z = -1) const;
};

inline double cubic(double x);
