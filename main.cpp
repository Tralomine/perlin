#include <SFML/Graphics.hpp>
#include <thread>
#include <cmath>
#include "sf_hsv_color.hpp"

#include "perlin.hpp"

int constexpr SizeX(1024), SizeY(1024);

void calcPerlin(Perlin* perlin, int x0, int y0, int xn, int yn, sf::Vertex * vertexes, int xv, int yv) {
  for (size_t x = x0; x < x0+xn; x++) {
    for (size_t y = y0; y < y0+yn; y++) {
      double c(1);
      double z = perlin[0].getPerlin(x, y) * 32;
      c += z/32. + perlin[1].getPerlin(x+173, y+57, z)/4.;
      if (c < 0.5) {
        c /= 5;
      } else if (c < 0.8) {
        c = (c-0.5) * 3 + 0.1;
      } else {
        c = (c-0.8) / 2. + 0.9;
      }
      // for (size_t i = 0; i < 6; i++) {
      //   c += perlin[i].getPerlin(x, y) / double(1<<(i+1));
      // }
      c /= 2;
      vertexes[x*SizeY+y] = {sf::Vector2f(xv+(x-x0), yv+(y-y0)), hsv((int)(c*500)*20, 0.75, 0.75)};
    }
  }
}

void calcPerlin3D(Perlin* perlin, int x0, int y0, int z, int xn, int yn, sf::Vertex * vertexes, int xv, int yv) {
  for (size_t x = x0; x < x0+xn; x++) {
    for (size_t y = y0; y < y0+yn; y++) {
      double c(0);
      // c = perlin[0].getPerlin(x, y, z);
      for (size_t i = 0; i < 3; i++) {
        c += perlin[i].getPerlin(x>SizeX/2?SizeX-x:x, y, z) / double(1<<(i+1));
      }
      c *= 256;
      // c -= 8;
      vertexes[x*SizeY+y] = {sf::Vector2f(xv+(x-x0), yv+(y-y0)), sf::Color(c, c, c)};
    }
  }
}

int main() {
  srand(time(NULL));

  sf::RenderWindow app(sf::VideoMode(SizeX, SizeY), "perlin", sf::Style::Close);
  // app.setPosition({0, 0});

  Perlin *perlin = new Perlin[6];
  // cudaMallocManaged(&perlin, 6*sizeof(Perlin));
  for (size_t i = 0; i < 6; i++) {
    perlin[i].setScale(256 / (1 << i));
  }

  app.clear(sf::Color::White);

  sf::Vertex *v = new sf::Vertex[SizeX*SizeY];
  // cudaMallocManaged(&v, SizeX*SizeY*sizeof(sf::Vertex));
  sf::VertexArray grid(sf::Lines);

  int z(0);

  std::thread* t = new std::thread[16];

  while (app.isOpen()) {
    sf::Event event;
    while (app.pollEvent(event)) {
      if (event.type == sf::Event::Closed) {
        app.close();
      }
    }

    // calcPerlin3D(perlin, 0, 0, z, SizeX, SizeY, v, 0, 0);
    for (size_t i = 0; i < 16; i++) {
      t[i] = std::thread(calcPerlin, perlin, i*SizeX/16, 0, SizeX/16, SizeY, v, i*(SizeX/16), 0);
    }
    
    for (size_t i = 0; i < 16; i++) {
      t[i].join();
    }

    // for (size_t x = 0; x < SizeX; x += SizeX/4) {
    //   for (size_t y = 0; y < SizeY; y += SizeY/4) {
    //     grid.append({{x, y}, sf::Color::Blue});
    //     grid.append({perlin->getVector(x, y)*128.f + sf::Vector2f(x, y), sf::Color::Red});
    //   }
    // }
    

    // z += 4;

    app.clear();
    app.draw(v, SizeX*SizeY, sf::Points);
    app.draw(grid);
    app.display();
  }

  delete v;
  delete perlin;
  // cudaFree(v);
  // cudaFree(perlin);

  return 0;
}
