// rime-qwen-freq: Standalone Rime Octagram (.gram) Inspector, Compiler, Typo Scanner & Hybrid Merger
// Build: c++ -O3 -std=c++17 tools/gram_tool.cc -o gram_tool
// Requires darts.h (Darts-clone v0.32, BSD license) in the same directory or include path.
//
// Key encoding follows upstream lotem/librime-octagram (src/gram_encoding.cc @ c030e30):
//   - U+0000            -> 0xE0
//   - U+0001..U+007F    -> 1 byte (ASCII)
//   - U+4000..U+9FFF    -> 2 bytes ((u >> 8) + 0x40, u & 0xFF), or (0xE1, (u >> 8) + 0x40) if low byte == 0
//   - Other codepoints  -> (0xE0 | n) followed by n 7-bit chunks (0x80 | chunk), left-aligned to bit 31 (u << 3)
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "darts.h"

struct GramHeader {
  char format[32];
  uint32_t db_checksum;
  uint32_t double_array_size;
  int32_t array_offset_rel;
};

static inline void strip_cr(std::string& s) {
  while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) {
    s.pop_back();
  }
}

static inline bool parse_int_safe(const std::string& s, int& out) {
  if (s.empty()) return false;
  char* endptr = nullptr;
  errno = 0;
  long v = std::strtol(s.c_str(), &endptr, 10);
  if (errno != 0 || endptr == s.c_str()) return false;
  while (*endptr == ' ' || *endptr == '\t' || *endptr == '\r') ++endptr;
  if (*endptr != '\0' || v < INT_MIN || v > INT_MAX) return false;
  out = static_cast<int>(v);
  return true;
}

static inline bool next_utf8_cp(const unsigned char*& p, const unsigned char* end, uint32_t& u) {
  if (p >= end) return false;
  unsigned char b0 = *p;
  if (b0 < 0x80) {
    u = b0;
    ++p;
    return true;
  }
  if ((b0 & 0xE0) == 0xC0) {
    if (p + 1 >= end || (p[1] & 0xC0) != 0x80) {
      ++p;
      return false;
    }
    u = ((uint32_t(b0 & 0x1F)) << 6) | uint32_t(p[1] & 0x3F);
    p += 2;
    return u >= 0x80;
  }
  if ((b0 & 0xF0) == 0xE0) {
    if (p + 2 >= end || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) {
      ++p;
      return false;
    }
    u = ((uint32_t(b0 & 0x0F)) << 12) | ((uint32_t(p[1] & 0x3F)) << 6) | uint32_t(p[2] & 0x3F);
    p += 3;
    return u >= 0x800 && !(u >= 0xD800 && u <= 0xDFFF);
  }
  if ((b0 & 0xF8) == 0xF0) {
    if (p + 3 >= end || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 || (p[3] & 0xC0) != 0x80) {
      ++p;
      return false;
    }
    u = ((uint32_t(b0 & 0x07)) << 18) | ((uint32_t(p[1] & 0x3F)) << 12) |
        ((uint32_t(p[2] & 0x3F)) << 6) | uint32_t(p[3] & 0x3F);
    p += 4;
    return u >= 0x10000 && u <= 0x10FFFF;
  }
  ++p;
  return false;
}

std::string encode_gram_key(const std::string& utf8) {
  std::string out;
  out.reserve(utf8.size());
  const unsigned char* p = reinterpret_cast<const unsigned char*>(utf8.data());
  const unsigned char* end = p + utf8.size();
  while (p < end) {
    uint32_t u = 0;
    if (!next_utf8_cp(p, end, u)) continue;

    if (u < 0x80) {
      out.push_back(u == 0 ? static_cast<char>(0xE0) : static_cast<char>(u));
    } else if (u >= 0x4000 && u < 0xA000) {
      if ((u & 0xFF) == 0) {
        out.push_back(static_cast<char>(0xE1));
        out.push_back(static_cast<char>((u >> 8) + 0x40));
      } else {
        out.push_back(static_cast<char>((u >> 8) + 0x40));
        out.push_back(static_cast<char>(u & 0xFF));
      }
    } else {
      int bits = 32;
      uint32_t tmp = u;
      while (bits > 0 && (tmp & 0xFE000000u) == 0) {
        bits -= 7;
        tmp <<= 7;
      }
      int bytes_to_encode = (bits + 6) / 7;
      out.push_back(static_cast<char>(0xE0 | bytes_to_encode));
      while (bytes_to_encode > 0) {
        --bytes_to_encode;
        out.push_back(static_cast<char>(((tmp >> 25) & 0x7F) | 0x80));
        tmp <<= 7;
      }
    }
  }
  return out;
}

bool decode_gram_to_utf8(const std::string& enc, std::string& utf8_out, std::vector<uint32_t>& cps) {
  utf8_out.clear();
  cps.clear();
  size_t i = 0;
  while (i < enc.size()) {
    unsigned char b = static_cast<unsigned char>(enc[i]);
    uint32_t u = 0;
    if (b < 0x80) {
      u = b;
      ++i;
    } else if (b >= 0x80 && b < 0xE0) {
      if (i + 1 >= enc.size()) return false;
      unsigned char b2 = static_cast<unsigned char>(enc[i + 1]);
      u = (uint32_t(b - 0x40) << 8) | uint32_t(b2);
      i += 2;
    } else if (b == 0xE0) {
      u = 0;
      ++i;
    } else if (b == 0xE1) {
      if (i + 1 >= enc.size()) return false;
      unsigned char b2 = static_cast<unsigned char>(enc[i + 1]);
      if (b2 < 0x40) return false;
      u = uint32_t(b2 - 0x40) << 8;
      i += 2;
    } else if (b >= 0xE2 && b <= 0xE5) {
      int n = b & 0x0F;
      if (i + static_cast<size_t>(n) >= enc.size()) return false;
      ++i;
      uint64_t acc = 0;
      for (int k = 0; k < n; ++k) {
        unsigned char bk = static_cast<unsigned char>(enc[i++]);
        if ((bk & 0x80) == 0) return false;
        acc = (acc << 7) | uint64_t(bk & 0x7F);
      }
      // encode_gram_key left-aligns 32-bit codepoints in 7-bit steps (bits = 32 - 7*k, n = 5 - k),
      // leaving 7*n - (32 - 7*k) = 35 - 32 = 3 trailing zero bits at the bottom.
      acc >>= 3;
      if (acc > 0x10FFFFULL) return false;
      u = static_cast<uint32_t>(acc);
    } else {
      return false;
    }

    if (u > 0x10FFFF || (u >= 0xD800 && u <= 0xDFFF)) return false;
    cps.push_back(u);
    if (u < 0x80) {
      utf8_out.push_back(char(u));
    } else if (u < 0x800) {
      utf8_out.push_back(char(0xC0 | (u >> 6)));
      utf8_out.push_back(char(0x80 | (u & 0x3F)));
    } else if (u < 0x10000) {
      utf8_out.push_back(char(0xE0 | (u >> 12)));
      utf8_out.push_back(char(0x80 | ((u >> 6) & 0x3F)));
      utf8_out.push_back(char(0x80 | (u & 0x3F)));
    } else {
      utf8_out.push_back(char(0xF0 | (u >> 18)));
      utf8_out.push_back(char(0x80 | ((u >> 12) & 0x3F)));
      utf8_out.push_back(char(0x80 | ((u >> 6) & 0x3F)));
      utf8_out.push_back(char(0x80 | (u & 0x3F)));
    }
  }
  return true;
}

static bool load_gram_da(
    const std::string& path,
    GramHeader& hdr,
    std::vector<char>& buf,
    Darts::DoubleArray& da) {
  std::ifstream fin(path, std::ios::binary | std::ios::ate);
  if (!fin) {
    std::cerr << "Error: cannot open .gram file: " << path << "\n";
    return false;
  }
  std::streamoff end_pos = fin.tellg();
  if (end_pos < static_cast<std::streamoff>(sizeof(GramHeader))) {
    std::cerr << "Error: .gram file too small: " << path << "\n";
    return false;
  }
  size_t file_size = static_cast<size_t>(end_pos);
  fin.seekg(0);
  if (!fin.read(reinterpret_cast<char*>(&hdr), sizeof(hdr))) {
    std::cerr << "Error: failed reading .gram header: " << path << "\n";
    return false;
  }
  if (std::strncmp(hdr.format, "Rime::Grammar/", 14) != 0) {
    std::cerr << "Error: invalid .gram magic header in " << path << "\n";
    return false;
  }
  int64_t array_start_s = static_cast<int64_t>(offsetof(GramHeader, array_offset_rel)) +
                          static_cast<int64_t>(hdr.array_offset_rel);
  if (array_start_s < static_cast<int64_t>(sizeof(GramHeader)) ||
      static_cast<uint64_t>(array_start_s) > file_size) {
    std::cerr << "Error: invalid array_offset_rel in " << path << "\n";
    return false;
  }
  size_t array_start = static_cast<size_t>(array_start_s);
  size_t array_bytes = static_cast<size_t>(hdr.double_array_size) * 4ULL;
  if (hdr.double_array_size == 0 || array_bytes > file_size - array_start) {
    std::cerr << "Error: truncated double-array payload in " << path << "\n";
    return false;
  }
  buf.resize(array_bytes);
  fin.seekg(static_cast<std::streamoff>(array_start));
  if (!fin.read(buf.data(), static_cast<std::streamsize>(array_bytes))) {
    std::cerr << "Error: failed reading double-array payload from " << path << "\n";
    return false;
  }
  da.set_array(buf.data(), hdr.double_array_size);
  return true;
}

bool write_gram_file(const std::string& out_path, const std::vector<std::pair<std::string, int>>& sorted_kv) {
  if (sorted_kv.empty()) {
    std::cerr << "Error: cannot build .gram from 0 valid entries\n";
    return false;
  }
  std::vector<const char*> keys(sorted_kv.size());
  std::vector<size_t> lengths(sorted_kv.size());
  std::vector<int> vals(sorted_kv.size());
  for (size_t i = 0; i < sorted_kv.size(); ++i) {
    keys[i] = sorted_kv[i].first.c_str();
    lengths[i] = sorted_kv[i].first.size();
    vals[i] = sorted_kv[i].second;
  }
  Darts::DoubleArray new_da;
  if (new_da.build(sorted_kv.size(), keys.data(), lengths.data(), vals.data()) != 0) {
    std::cerr << "Error building Darts DoubleArray!\n";
    return false;
  }
  GramHeader out_hdr{};
  std::memset(&out_hdr, 0, sizeof(out_hdr));
  std::strncpy(out_hdr.format, "Rime::Grammar/1.0", 31);
  out_hdr.db_checksum = 0;
  out_hdr.double_array_size = static_cast<uint32_t>(new_da.size());
  out_hdr.array_offset_rel = 4;

  std::ofstream fout(out_path, std::ios::binary);
  if (!fout) return false;
  fout.write(reinterpret_cast<const char*>(&out_hdr), sizeof(out_hdr));
  fout.write(reinterpret_cast<const char*>(new_da.array()), new_da.total_size());
  fout.close();
  double mib = (sizeof(out_hdr) + new_da.total_size()) / (1024.0 * 1024.0);
  std::cerr << "Built " << out_path << " | keys=" << sorted_kv.size()
            << " | units=" << new_da.size()
            << " | size=" << mib << " MiB\n";
  return true;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage:\n"
              << "  gram_tool query <file.gram> <queries.txt> <out.tsv>\n"
              << "  gram_tool build <scaled_entries.tsv> <out.gram>\n"
              << "  gram_tool merge <base.gram> <blacklist.txt|-> <additions.tsv> <min_keep_val> <out.gram>\n"
              << "  gram_tool scan_typos <base.gram> <char_fuzzy_py.tsv> <base_words.txt> <out_suspects.tsv>\n";
    return 1;
  }
  std::string cmd = argv[1];

  if (cmd == "query" && argc >= 5) {
    GramHeader hdr{};
    std::vector<char> buf;
    Darts::DoubleArray da;
    if (!load_gram_da(argv[2], hdr, buf, da)) return 1;

    std::ifstream fq(argv[3]);
    std::ofstream fo(argv[4]);
    if (!fq || !fo) return 1;
    std::string line;
    while (std::getline(fq, line)) {
      strip_cr(line);
      if (line.empty()) continue;
      std::string enc = encode_gram_key(line);
      int v = enc.empty() ? -1 : da.exactMatchSearch<int>(enc.c_str(), enc.size());
      fo << line << "\t" << v << "\n";
    }
    return 0;
  }

  if ((cmd == "build" || cmd == "build_pure") && argc >= 4) {
    std::ifstream fin(argv[2]);
    if (!fin) return 1;
    std::vector<std::pair<std::string, int>> kv;
    std::string line;
    while (std::getline(fin, line)) {
      strip_cr(line);
      if (line.empty() || line[0] == '#') continue;
      size_t tab = line.find('\t');
      if (tab == std::string::npos) continue;
      std::string k = line.substr(0, tab);
      int v = 0;
      if (!parse_int_safe(line.substr(tab + 1), v)) continue;
      if (k.empty() || v <= 0) continue;
      std::string enc = encode_gram_key(k);
      if (enc.empty() || enc.find('\0') != std::string::npos) continue;
      kv.emplace_back(std::move(enc), v);
    }
    std::sort(kv.begin(), kv.end(), [](const auto& a, const auto& b) {
      int c = std::strcmp(a.first.c_str(), b.first.c_str());
      if (c != 0) return c < 0;
      return a.second > b.second;
    });
    size_t w = 0;
    for (size_t r = 0; r < kv.size(); ++r) {
      if (w == 0 || kv[r].first != kv[w - 1].first) {
        if (w != r) kv[w] = std::move(kv[r]);
        ++w;
      }
    }
    kv.resize(w);
    return write_gram_file(argv[3], kv) ? 0 : 1;
  }

  if ((cmd == "merge" || cmd == "build_hybrid") && argc >= 7) {
    std::string base_gram = argv[2];
    std::string blk_path = argv[3];
    std::string add_path = argv[4];
    int min_keep = 0;
    if (!parse_int_safe(argv[5], min_keep)) {
      std::cerr << "Error: invalid integer min_keep_val: " << argv[5] << "\n";
      return 1;
    }
    std::string out_path = argv[6];

    std::unordered_set<std::string> blk_enc;
    if (blk_path != "-" && !blk_path.empty()) {
      std::ifstream fb(blk_path);
      std::string line;
      while (std::getline(fb, line)) {
        strip_cr(line);
        if (!line.empty()) blk_enc.insert(encode_gram_key(line));
      }
    }

    std::vector<std::pair<std::string, int>> add_kv;
    {
      std::ifstream fa(add_path);
      if (!fa) return 1;
      std::string line;
      while (std::getline(fa, line)) {
        strip_cr(line);
        if (line.empty() || line[0] == '#') continue;
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string k = line.substr(0, tab);
        int v = 0;
        if (!parse_int_safe(line.substr(tab + 1), v)) continue;
        if (k.empty() || v <= 0) continue;
        std::string enc = encode_gram_key(k);
        if (enc.empty() || enc.find('\0') != std::string::npos) continue;
        add_kv.emplace_back(std::move(enc), v);
      }
    }

    GramHeader hdr{};
    std::vector<char> buf;
    Darts::DoubleArray da;
    if (!load_gram_da(base_gram, hdr, buf, da)) return 1;

    std::vector<std::pair<std::string, int>> merged;
    size_t est_keys = std::min<size_t>(
        55000000ULL,
        static_cast<size_t>(hdr.double_array_size) / 2ULL + add_kv.size() + 1024ULL);
    merged.reserve(est_keys);

    struct Frame { size_t node_pos; int next_byte; };
    std::string cur_key;
    std::vector<Frame> st = {{0, 1}};
    while (!st.empty()) {
      Frame& top = st.back();
      bool moved = false;
      while (top.next_byte < 256) {
        char ch = static_cast<char>(top.next_byte++);
        size_t npos = top.node_pos, kpos = 0;
        int res = da.traverse(&ch, npos, kpos, 1);
        if (res != -2) {
          cur_key.push_back(ch);
          if (res >= 0 && blk_enc.find(cur_key) == blk_enc.end() && res >= min_keep) {
            merged.emplace_back(cur_key, res);
          }
          st.push_back({npos, 1});
          moved = true;
          break;
        }
      }
      if (!moved) {
        st.pop_back();
        if (!cur_key.empty()) cur_key.pop_back();
      }
    }
    buf.clear();
    buf.shrink_to_fit();

    for (auto& kv : add_kv) merged.emplace_back(std::move(kv.first), kv.second);
    add_kv.clear();
    add_kv.shrink_to_fit();

    std::sort(merged.begin(), merged.end(), [](const auto& a, const auto& b) {
      int c = std::strcmp(a.first.c_str(), b.first.c_str());
      if (c != 0) return c < 0;
      return a.second > b.second;
    });
    size_t mw = 0;
    for (size_t r = 0; r < merged.size(); ++r) {
      if (mw == 0 || merged[r].first != merged[mw - 1].first) {
        if (mw != r) merged[mw] = std::move(merged[r]);
        ++mw;
      }
    }
    merged.resize(mw);
    return write_gram_file(out_path, merged) ? 0 : 1;
  }

  if (cmd == "scan_typos" && argc >= 6) {
    std::unordered_map<uint32_t, uint16_t> cp_to_fpy;
    std::unordered_map<std::string, uint16_t> fpy_vocab;
    {
      std::ifstream fc(argv[3]);
      if (!fc) return 1;
      std::string line;
      while (std::getline(fc, line)) {
        strip_cr(line);
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string ch = line.substr(0, tab);
        std::string fpy = line.substr(tab + 1);
        const unsigned char* p = reinterpret_cast<const unsigned char*>(ch.data());
        const unsigned char* end = p + ch.size();
        uint32_t u = 0;
        if (!next_utf8_cp(p, end, u) || fpy.empty()) continue;
        auto it = fpy_vocab.find(fpy);
        uint16_t id = (it == fpy_vocab.end()) ? (fpy_vocab[fpy] = uint16_t(fpy_vocab.size() + 1)) : it->second;
        cp_to_fpy[u] = id;
      }
    }
    auto pack_fpy = [&](const std::vector<uint32_t>& cps) -> uint64_t {
      if (cps.size() < 3 || cps.size() > 6) return 0;
      uint64_t h = uint64_t(cps.size());
      for (uint32_t u : cps) {
        auto it = cp_to_fpy.find(u);
        if (it == cp_to_fpy.end()) return 0;
        h = (h << 10) ^ uint64_t(it->second);
      }
      return h;
    };
    std::unordered_set<std::string> base_set;
    std::unordered_map<uint64_t, std::vector<std::string>> fpy_to_base;
    {
      std::ifstream fb(argv[4]);
      if (!fb) return 1;
      std::string w;
      while (std::getline(fb, w)) {
        strip_cr(w);
        if (w.empty()) continue;
        base_set.insert(w);
        std::vector<uint32_t> cps;
        const unsigned char* p = reinterpret_cast<const unsigned char*>(w.data());
        const unsigned char* end = p + w.size();
        while (p < end) {
          uint32_t u = 0;
          if (next_utf8_cp(p, end, u)) cps.push_back(u);
        }
        uint64_t h = pack_fpy(cps);
        if (h != 0) fpy_to_base[h].push_back(w);
      }
    }

    GramHeader hdr{};
    std::vector<char> buf;
    Darts::DoubleArray da;
    if (!load_gram_da(argv[2], hdr, buf, da)) return 1;

    std::ofstream fout(argv[5]);
    if (!fout) return 1;
    struct Frame { size_t node_pos; int next_byte; };
    std::string cur_key, utf8_s;
    std::vector<uint32_t> cps;
    std::vector<Frame> st = {{0, 1}};
    while (!st.empty()) {
      Frame& top = st.back();
      bool moved = false;
      while (top.next_byte < 256) {
        char ch = static_cast<char>(top.next_byte++);
        size_t npos = top.node_pos, kpos = 0;
        int res = da.traverse(&ch, npos, kpos, 1);
        if (res != -2) {
          cur_key.push_back(ch);
          if (res >= 158000 && decode_gram_to_utf8(cur_key, utf8_s, cps) && cps.size() >= 3 && cps.size() <= 6) {
            if (base_set.find(utf8_s) == base_set.end()) {
              uint64_t h = pack_fpy(cps);
              auto it = fpy_to_base.find(h);
              if (it != fpy_to_base.end()) {
                fout << utf8_s << "\t" << res << "\t";
                for (size_t k = 0; k < it->second.size() && k < 4; ++k) {
                  if (k) fout << "|";
                  fout << it->second[k];
                }
                fout << "\n";
              }
            }
          }
          st.push_back({npos, 1});
          moved = true;
          break;
        }
      }
      if (!moved) {
        st.pop_back();
        if (!cur_key.empty()) cur_key.pop_back();
      }
    }
    return 0;
  }

  return 1;
}
