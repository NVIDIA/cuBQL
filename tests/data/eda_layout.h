// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/math/box.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace eda_layout {

enum class Pattern { Layout, Motif };

struct Rectangle { int x0, y0, x1, y1; };

// Original integer geometry: fingers, routing rails, vertical routes, contacts,
// and a concentric pair that deliberately produces equal centroid keys.
inline constexpr Rectangle motif[] = {
  { 8, 12, 12, 36}, {20, 12, 24, 36}, {32, 12, 36, 36}, {44, 12, 48, 36},
  {56, 12, 60, 36}, {68, 12, 72, 36}, {80, 12, 84, 36}, {92, 12, 96, 36},
  { 8, 72, 12, 96}, {20, 72, 24, 96}, {32, 72, 36, 96}, {44, 72, 48, 96},
  {56, 72, 60, 96}, {68, 72, 72, 96}, {80, 72, 84, 96}, {92, 72, 96, 96},
  { 4,  4,100,  8}, { 4, 40,100, 44}, { 4, 64,100, 68}, { 4,100,100,104},
  { 2,  4,  4,104}, {104,  4,106,104}, {16, 44, 18, 64}, {88, 44, 90, 64},
  { 8, 46, 14, 52}, {26, 46, 32, 52}, {44, 46, 50, 52}, {62, 46, 68, 52},
  {80, 46, 86, 52}, {98, 46,104, 52}, {48, 54, 56, 62}, {50, 56, 54, 60}
};
inline constexpr uint32_t rectanglesPerTile = sizeof(motif)/sizeof(motif[0]);
inline constexpr int tilePitch = 128;

inline Rectangle layoutRectangle(uint32_t x, uint32_t y, uint32_t r)
{
  const uint32_t groupX = x/3;
  const uint32_t groupY = y/2;
  const uint32_t selection = (5*groupX+7*groupY+3*(groupX^groupY))%19;
  uint32_t kind = selection < 8 ? 0 : selection < 13 ? 1 : selection < 18 ? 2 : 3;
  if ((x%11 == 5 && y%5 != 1) || (y%9 == 4 && x%7 != 2)) kind = 3;
  Rectangle rect;
  if (kind == 0) {
    if (r < 16) {
      const int x0 = 8+14*int(r%8);
      const int y0 = 8+72*int(r/8);
      rect = {x0,y0,x0+10,y0+34};
    } else if (r < 20) {
      const int i = int(r-16);
      const int y0 = 4+72*(i/2)+38*(i%2);
      rect = {2,y0,122,y0+4};
    } else if (r < 28) {
      const int x0 = 10+14*int(r-20);
      rect = {x0,54,x0+8,62};
    } else if (r < 30) {
      const int x0 = 2+118*int(r-28);
      rect = {x0,4,x0+4,118};
    } else {
      rect = r == 30 ? Rectangle{56,64,72,74} : Rectangle{58,66,70,72};
    }
  } else if (kind == 1) {
    if (r < 16) {
      const int group = int(r/4);
      const int x0 = 8+60*(group%2)+12*int(r%4);
      const int y0 = 8+60*(group/2);
      rect = {x0,y0,x0+8,y0+28};
    } else if (r < 24) {
      const int i = int(r-16);
      const int x0 = 4+60*((i/2)%2);
      const int y0 = 4+60*(i/4)+32*(i%2);
      rect = {x0,y0,x0+52,y0+4};
    } else if (r < 28) {
      const int i = int(r-24);
      const int x0 = 4+60*(i%2);
      const int y0 = 4+60*(i/2);
      rect = {x0,y0,x0+4,y0+36};
    } else if (r < 30) {
      const int y0 = 48+72*int(r-28);
      rect = {2,y0,124,y0+4};
    } else {
      rect = r == 30 ? Rectangle{56,48,72,60} : Rectangle{58,50,70,58};
    }
  } else if (kind == 2) {
    if (r < 16) {
      const int i = int(r/2);
      const int x0 = 2+30*(i%4);
      const int y0 = 8+60*(i/4);
      rect = r%2 == 0 ? Rectangle{x0,y0,x0+24,y0+10}
                      : Rectangle{x0+16,y0+10,x0+24,y0+38};
    } else if (r < 24) {
      const int i = int(r-16);
      const int x0 = 8+30*(i%4);
      const int y0 = 28+60*(i/4);
      rect = {x0,y0,x0+8,y0+8};
    } else if (r < 28) {
      const int i = int(r-24);
      const int y0 = 4+56*(i/2)+4*(i%2);
      rect = {4,y0,124,y0+3};
    } else if (r < 30) {
      const int x0 = 4+116*int(r-28);
      rect = {x0,4,x0+4,120};
    } else {
      rect = r == 30 ? Rectangle{56,50,72,58} : Rectangle{58,52,70,56};
    }
  } else {
    if (r < 16) {
      const int i = int(r/2);
      const int x0 = 8+72*(i%2);
      const int y0 = 8+28*(i/2);
      rect = r%2 == 0 ? Rectangle{x0,y0,x0+36,y0+6}
                      : Rectangle{x0+28,y0+6,x0+36,y0+20};
    } else if (r < 28) {
      const int i = int(r-16);
      const int x0 = 12+12*(i%3)+72*(i/6);
      const int y0 = 30+32*((i%6)/3);
      rect = {x0,y0,x0+6,y0+6};
    } else if (r < 30) {
      const int x0 = 2+120*int(r-28);
      rect = {x0,4,x0+4,124};
    } else {
      rect = r == 30 ? Rectangle{58,58,70,70} : Rectangle{60,60,68,68};
    }
  }
  if ((groupX+groupY)%2) {
    const int x0 = tilePitch-rect.x1;
    rect.x1 = tilePitch-rect.x0;
    rect.x0 = x0;
  }
  if ((x/5+y/3)%2) {
    const int y0 = tilePitch-rect.y1;
    rect.y1 = tilePitch-rect.y0;
    rect.y0 = y0;
  }
  return rect;
}

template<typename T, int D>
std::vector<cuBQL::box_t<T,D>> makeBoxes(uint32_t tilesX, uint32_t tilesY,
                                      Pattern pattern=Pattern::Layout)
{
  static_assert(D >= 2 && D <= 4);
  if (tilesX == 0 || tilesY == 0 || tilesX > 131072 || tilesY > 131072)
    throw std::invalid_argument("Synthetic layout requires 1 to 131072 tiles per axis");
  if (uint64_t(tilesX)*tilesY*rectanglesPerTile > uint64_t(std::numeric_limits<int>::max())/2)
    throw std::invalid_argument("Synthetic layout exceeds the radix builder node limit");
  std::vector<cuBQL::box_t<T,D>> boxes(std::size_t(tilesX)*tilesY*rectanglesPerTile);
  std::size_t index = 0;
  for (uint32_t y=0;y<tilesY;++y)
    for (uint32_t x=0;x<tilesX;++x)
      for (uint32_t r=0;r<rectanglesPerTile;++r) {
        const Rectangle rect = pattern == Pattern::Motif ? motif[r] : layoutRectangle(x,y,r);
        const int dx = int(x)*tilePitch-tilePitch;
        const int dy = int(y)*tilePitch-tilePitch;
        auto &box = boxes[index++];
        box.lower[0] = T(dx+rect.x0); box.upper[0] = T(dx+rect.x1);
        box.lower[1] = T(dy+rect.y0); box.upper[1] = T(dy+rect.y1);
        for (int axis=2;axis<D;++axis) {
          const int lower = axis == 2 ? 4*int(r%4)-8 : 4*int(r/4)-16;
          box.lower[axis] = T(lower);
          box.upper[axis] = T(lower+2);
        }
      }
  return boxes;
}

// Canonical little-endian signed integer coordinates, independent of scalar type.
template<typename T, int D>
uint64_t fingerprint(const std::vector<cuBQL::box_t<T,D>> &boxes)
{
  uint64_t hash = UINT64_C(14695981039346656037);
  for (const auto &box : boxes)
    for (int axis=0;axis<D;++axis)
      for (int side=0;side<2;++side) {
        uint64_t value = uint64_t(int64_t(side ? box.upper[axis] : box.lower[axis]));
        for (int byte=0;byte<8;++byte) {
          hash ^= (value >> (8*byte)) & 255u;
          hash *= UINT64_C(1099511628211);
        }
      }
  return hash;
}

inline void writeSvg(const char *path, uint32_t tilesX=16, uint32_t tilesY=8,
                     Pattern pattern=Pattern::Layout)
{
  if (!path || !*path)
    throw std::invalid_argument("SVG output path must not be empty");
  if (tilesX == 0 || tilesY == 0 || tilesX > 32 || tilesY > 32)
    throw std::invalid_argument("SVG preview requires 1 to 32 tiles per axis");
  const auto boxes = makeBoxes<float,2>(tilesX,tilesY,pattern);
  auto bounds = boxes.front();
  for (const auto &box : boxes)
    bounds.grow(box);
  const int left = int(bounds.lower[0])-18;
  const int top = int(bounds.lower[1])-46;
  const int width = std::max(340,int(bounds.upper[0]-bounds.lower[0])+36);
  const int height = int(bounds.upper[1]-bounds.lower[1])+64;
  std::ofstream out;
  out.exceptions(std::ios::failbit | std::ios::badbit);
  try {
    out.open(path);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\""
        << left << ' ' << top << ' ' << width << ' ' << height << "\">\n"
        << "<title>Original synthetic layout</title>\n"
        << "<rect x=\"" << left << "\" y=\"" << top
        << "\" width=\"" << width << "\" height=\"" << height
        << "\" fill=\"#000000\"/>\n"
        << "<text x=\"" << int(bounds.lower[0]) << "\" y=\""
        << int(bounds.lower[1])-28
        << "\" fill=\"#dbe5eb\" font-family=\"sans-serif\" font-size=\"14\">"
        << "Original synthetic layout: " << boxes.size() << " rectangles</text>\n";
    for (std::size_t i=0;i<boxes.size();++i) {
      const auto &box = boxes[i];
      const uint32_t r = uint32_t(i%rectanglesPerTile);
      const char *color = pattern == Pattern::Layout ? "#72b59f"
                          : (r >= 16 && r < 24) || r == 31 ? "#52c7bd" : "#e7be62";
      out << "<rect x=\"" << int(box.lower[0]) << "\" y=\"" << int(box.lower[1])
          << "\" width=\"" << int(box.upper[0]-box.lower[0])
          << "\" height=\"" << int(box.upper[1]-box.lower[1])
          << "\" fill=\"" << color << "\"/>\n";
    }
    out << "</svg>\n";
    out.close();
  } catch (const std::ios_base::failure &) {
    throw std::runtime_error(std::string("Failed to write SVG preview: ")+path);
  }
}

} // namespace eda_layout
