// Native C++ End-to-End Rime + Octagram Benchmark Runner
// Dynamically loads librime (+ optional octagram & lua plugins across macOS, Linux, and Windows)
// and evaluates real Pinyin -> Chinese full-sentence, contextual, and phrase ranking accuracy.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "rime_api.h"

namespace fs = std::filesystem;

namespace {

#ifdef _WIN32
using LibHandle = HMODULE;
LibHandle open_dynlib(const std::string& path) { return LoadLibraryA(path.c_str()); }
void* find_dynsym(LibHandle h, const char* sym) { return reinterpret_cast<void*>(GetProcAddress(h, sym)); }
std::string last_dynerr() { return "LoadLibrary/GetProcAddress error code " + std::to_string(GetLastError()); }
#else
using LibHandle = void*;
LibHandle open_dynlib(const std::string& path) { return dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL); }
void* find_dynsym(LibHandle h, const char* sym) { return dlsym(h, sym); }
std::string last_dynerr() {
  const char* e = dlerror();
  return e ? std::string(e) : "unknown dlopen error";
}
#endif

std::string env_or_empty(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : std::string();
}

void strip_cr(std::string& s) {
  while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
}

std::vector<std::string> candidate_rime_libs(const std::string& explicit_path) {
  if (!explicit_path.empty()) return {explicit_path};
  std::string from_env = env_or_empty("RIME_LIB_PATH");
  if (!from_env.empty()) return {from_env};

  std::vector<std::string> cands;
#ifdef __APPLE__
  cands = {
      "/Library/Input Methods/Squirrel.app/Contents/Frameworks/librime.1.dylib",
      "/Library/Input Methods/Squirrel.app/Contents/Frameworks/librime.dylib",
      "/opt/homebrew/lib/librime.1.dylib",
      "/opt/homebrew/lib/librime.dylib",
      "/usr/local/lib/librime.1.dylib",
      "/usr/local/lib/librime.dylib",
      "librime.1.dylib",
      "librime.dylib",
  };
#elif defined(_WIN32)
  for (const char* env_key : {"ProgramFiles(x86)", "ProgramFiles"}) {
    std::string pf = env_or_empty(env_key);
    if (!pf.empty()) {
      fs::path rime_base = fs::path(pf) / "Rime";
      std::error_code ec;
      if (fs::is_directory(rime_base, ec)) {
        for (const auto& entry : fs::directory_iterator(rime_base, ec)) {
          fs::path dll = entry.path() / "rime.dll";
          if (fs::is_regular_file(dll, ec)) cands.push_back(dll.string());
        }
      }
    }
  }
  cands.push_back("rime.dll");
#else
  cands = {
      "librime.so.1",
      "librime.so",
      "/usr/lib/x86_64-linux-gnu/librime.so.1",
      "/usr/lib/aarch64-linux-gnu/librime.so.1",
      "/usr/lib64/librime.so.1",
      "/usr/lib/librime.so.1",
      "/usr/local/lib/librime.so.1",
  };
#endif
  return cands;
}

void try_load_plugin(
    const std::string& explicit_path,
    const char* env_var,
    const std::string& rime_lib_loaded,
    const std::vector<std::string>& names) {
  std::string target = !explicit_path.empty() ? explicit_path : env_or_empty(env_var);
  if (!target.empty()) {
    if (!open_dynlib(target)) {
      std::cerr << "Warning: failed to load plugin " << target << ": " << last_dynerr() << "\n";
    }
    return;
  }

  std::vector<fs::path> search_dirs;
  std::error_code ec;
  fs::path rime_parent = fs::path(rime_lib_loaded).parent_path();
  if (!rime_parent.empty()) {
    search_dirs.push_back(rime_parent / "rime-plugins");
    search_dirs.push_back(rime_parent);
  }
#ifdef __APPLE__
  search_dirs.emplace_back("/Library/Input Methods/Squirrel.app/Contents/Frameworks/rime-plugins");
  search_dirs.emplace_back("/opt/homebrew/lib/rime-plugins");
  search_dirs.emplace_back("/usr/local/lib/rime-plugins");
#elif !defined(_WIN32)
  search_dirs.emplace_back("/usr/lib/x86_64-linux-gnu/rime-plugins");
  search_dirs.emplace_back("/usr/lib/aarch64-linux-gnu/rime-plugins");
  search_dirs.emplace_back("/usr/lib64/rime-plugins");
  search_dirs.emplace_back("/usr/lib/rime-plugins");
  search_dirs.emplace_back("/usr/local/lib/rime-plugins");
#endif

  for (const auto& dir : search_dirs) {
    for (const auto& nm : names) {
      fs::path p = dir / nm;
      if (fs::is_regular_file(p, ec)) {
        if (open_dynlib(p.string())) return;
      }
    }
  }
  for (const auto& nm : names) {
    if (open_dynlib(nm)) return;
  }
}

std::string default_shared_data_dir() {
#ifdef __APPLE__
  for (const char* p : {
           "/Library/Input Methods/Squirrel.app/Contents/SharedSupport",
           "/opt/homebrew/share/rime-data",
           "/usr/local/share/rime-data"}) {
    std::error_code ec;
    if (fs::is_directory(p, ec)) return p;
  }
#elif defined(_WIN32)
  for (const char* env_key : {"ProgramFiles(x86)", "ProgramFiles"}) {
    std::string pf = env_or_empty(env_key);
    if (!pf.empty()) {
      fs::path rime_base = fs::path(pf) / "Rime";
      std::error_code ec;
      if (fs::is_directory(rime_base, ec)) {
        for (const auto& entry : fs::directory_iterator(rime_base, ec)) {
          fs::path data = entry.path() / "data";
          if (fs::is_directory(data, ec)) return data.string();
        }
      }
    }
  }
#else
  for (const char* p : {"/usr/share/rime-data", "/usr/local/share/rime-data"}) {
    std::error_code ec;
    if (fs::is_directory(p, ec)) return p;
  }
#endif
  return "";
}

std::string json_escape(const std::string& s) {
  std::ostringstream o;
  for (unsigned char c : s) {
    switch (c) {
      case '"': o << "\\\""; break;
      case '\\': o << "\\\\"; break;
      case '\b': o << "\\b"; break;
      case '\f': o << "\\f"; break;
      case '\n': o << "\\n"; break;
      case '\r': o << "\\r"; break;
      case '\t': o << "\\t"; break;
      default:
        if (c < 0x20) {
          o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        } else {
          o << char(c);
        }
    }
  }
  return o.str();
}

std::vector<uint32_t> to_codepoints(const std::string& s) {
  std::vector<uint32_t> cps;
  const unsigned char* p = reinterpret_cast<const unsigned char*>(s.data());
  const unsigned char* end = p + s.size();
  while (p < end) {
    uint32_t u = 0;
    if (*p < 0x80) { u = *p++; }
    else if ((*p & 0xE0) == 0xC0 && p + 1 < end) { u = ((*p & 0x1F) << 6) | (p[1] & 0x3F); p += 2; }
    else if ((*p & 0xF0) == 0xE0 && p + 2 < end) { u = ((*p & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F); p += 3; }
    else if ((*p & 0xF8) == 0xF0 && p + 3 < end) { u = ((*p & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F); p += 4; }
    else { ++p; continue; }
    cps.push_back(u);
  }
  return cps;
}

int edit_distance(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
  size_t n = a.size(), m = b.size();
  std::vector<int> dp(m + 1);
  for (size_t j = 0; j <= m; ++j) dp[j] = int(j);
  for (size_t i = 1; i <= n; ++i) {
    int prev = dp[0];
    dp[0] = int(i);
    for (size_t j = 1; j <= m; ++j) {
      int cur = dp[j];
      if (a[i - 1] == b[j - 1]) dp[j] = prev;
      else dp[j] = 1 + std::min({prev, dp[j], dp[j - 1]});
      prev = cur;
    }
  }
  return dp[m];
}

}  // namespace

int main(int argc, char** argv) {
  std::string user_dir;
  std::string shared_dir = default_shared_data_dir();
  std::string cases_path;
  std::string out_json;
  std::string schema_id = "rime_mint";
  std::string rime_lib_arg;
  std::string octa_lib_arg;
  std::string lua_lib_arg;
  bool do_deploy = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--user-dir" && i + 1 < argc) user_dir = argv[++i];
    else if (a == "--shared-dir" && i + 1 < argc) shared_dir = argv[++i];
    else if (a == "--cases" && i + 1 < argc) cases_path = argv[++i];
    else if (a == "--out" && i + 1 < argc) out_json = argv[++i];
    else if (a == "--schema" && i + 1 < argc) schema_id = argv[++i];
    else if (a == "--rime-lib" && i + 1 < argc) rime_lib_arg = argv[++i];
    else if (a == "--octagram-lib" && i + 1 < argc) octa_lib_arg = argv[++i];
    else if (a == "--lua-lib" && i + 1 < argc) lua_lib_arg = argv[++i];
    else if (a == "--deploy") do_deploy = true;
  }

  if (user_dir.empty() || cases_path.empty()) {
    std::cerr << "Usage: rime_bench --user-dir <dir> --cases <tsv> [--out <json>] [--schema rime_mint] "
              << "[--shared-dir <dir>] [--rime-lib <path>] [--octagram-lib <path>] [--lua-lib <path>] [--deploy]\n";
    return 1;
  }
  if (shared_dir.empty()) shared_dir = user_dir;

  LibHandle h_rime = nullptr;
  std::string loaded_rime_path;
  for (const auto& cand : candidate_rime_libs(rime_lib_arg)) {
    h_rime = open_dynlib(cand);
    if (h_rime) {
      loaded_rime_path = cand;
      break;
    }
  }
  if (!h_rime) {
    std::cerr << "Error: could not load librime (" << last_dynerr()
              << "). Pass --rime-lib or set RIME_LIB_PATH.\n";
    return 1;
  }

  try_load_plugin(
      octa_lib_arg,
      "RIME_OCTAGRAM_LIB_PATH",
      loaded_rime_path,
      {"librime-octagram.dylib", "librime-octagram.so", "rime-octagram.dll", "octagram.dll"});
  try_load_plugin(
      lua_lib_arg,
      "RIME_LUA_LIB_PATH",
      loaded_rime_path,
      {"librime-lua.dylib", "librime-lua.so", "rime-lua.dll", "lua.dll"});

  using GetApiFn = RimeApi* (*)(void);
  auto get_api = reinterpret_cast<GetApiFn>(find_dynsym(h_rime, "rime_get_api"));
  if (!get_api) {
    std::cerr << "Error: symbol rime_get_api not found in " << loaded_rime_path << "\n";
    return 1;
  }
  RimeApi* api = get_api();

  std::vector<const char*> modules_vec = {"default"};
  if (RIME_API_AVAILABLE(api, find_module)) {
    if (api->find_module("octagram")) modules_vec.push_back("octagram");
    if (api->find_module("lua")) modules_vec.push_back("lua");
  } else {
    modules_vec.push_back("octagram");
    modules_vec.push_back("lua");
  }
  modules_vec.push_back(nullptr);

  RIME_STRUCT(RimeTraits, traits);
  traits.shared_data_dir = shared_dir.c_str();
  traits.user_data_dir = user_dir.c_str();
  traits.distribution_name = "RimeBench";
  traits.distribution_code_name = "rime-bench";
  traits.distribution_version = "1.0";
  traits.app_name = "rime.bench";
  traits.modules = modules_vec.data();
  traits.min_log_level = 2; // ERROR only
  traits.log_dir = "";

  api->setup(&traits);
  api->initialize(&traits);

  if (do_deploy) {
    if (api->start_maintenance(True)) {
      api->join_maintenance_thread();
    }
  }

  // Load TSV cases: category \t mode \t context_pinyin_or_empty \t context_expected_or_empty \t pinyin \t expected
  std::ifstream fc(cases_path);
  if (!fc) {
    std::cerr << "Cannot open cases file: " << cases_path << "\n";
    api->finalize();
    return 1;
  }

  struct FullCase {
    std::string category;
    std::string mode;
    std::string ctx_pinyin;
    std::string ctx_text;
    std::string pinyin;
    std::string expected;
  };
  std::vector<FullCase> cases;
  std::string line;
  while (std::getline(fc, line)) {
    strip_cr(line);
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string col;
    while (std::getline(ss, col, '\t')) cols.push_back(col);
    if (cols.size() >= 6) {
      cases.push_back({cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]});
    }
  }

  RimeSessionId sess = api->create_session();
  api->select_schema(sess, schema_id.c_str());
  api->set_option(sess, "transcription", False);
  api->set_option(sess, "ascii_mode", False);

  int total = 0, top1_hits = 0, top3_hits = 0, top5_hits = 0;
  double sum_mrr = 0.0;
  size_t total_chars = 0, total_char_edits = 0;

  struct CaseResult {
    FullCase c;
    std::string top1;
    int found_rank; // 1-based, 0 if not in top 50
    double char_acc;
    std::vector<std::string> top5;
  };
  std::vector<CaseResult> results;

  for (const auto& c : cases) {
    api->clear_composition(sess);
    {
      RIME_STRUCT(RimeCommit, cm);
      if (api->get_commit(sess, &cm)) api->free_commit(&cm);
    }

    if (c.mode == "contextual" && !c.ctx_pinyin.empty()) {
      api->simulate_key_sequence(sess, c.ctx_pinyin.c_str());
      RimeCandidateListIterator it{};
      int target_idx = 0;
      if (api->candidate_list_begin(sess, &it)) {
        while (api->candidate_list_next(&it)) {
          if (it.candidate.text && c.ctx_text == it.candidate.text) {
            target_idx = it.index;
            break;
          }
          if (it.index >= 20) break;
        }
        api->candidate_list_end(&it);
      }
      if (RIME_API_AVAILABLE(api, select_candidate_on_current_page) && target_idx == 0) {
        api->commit_composition(sess);
      } else if (RIME_API_AVAILABLE(api, select_candidate)) {
        if (!api->select_candidate(sess, target_idx)) {
          api->commit_composition(sess);
        }
      } else {
        api->commit_composition(sess);
      }
      RIME_STRUCT(RimeCommit, cm);
      if (api->get_commit(sess, &cm)) api->free_commit(&cm);
    }

    api->simulate_key_sequence(sess, c.pinyin.c_str());

    std::string top1;
    int rank = 0;
    std::vector<std::string> top5;

    RimeCandidateListIterator iter{};
    if (api->candidate_list_begin(sess, &iter)) {
      while (api->candidate_list_next(&iter)) {
        std::string cand = iter.candidate.text ? iter.candidate.text : "";
        if (iter.index == 0) top1 = cand;
        if (top5.size() < 5) top5.push_back(cand);
        if (rank == 0 && cand == c.expected) {
          rank = iter.index + 1;
        }
        if (iter.index >= 49) break;
      }
      api->candidate_list_end(&iter);
    }

    if (top1.empty()) {
      RIME_STRUCT(RimeContext, ctx);
      if (api->get_context(sess, &ctx)) {
        if (ctx.commit_text_preview) top1 = ctx.commit_text_preview;
        if (top1 == c.expected && rank == 0) rank = 1;
        api->free_context(&ctx);
      }
    }

    auto cp_exp = to_codepoints(c.expected);
    auto cp_got = to_codepoints(top1);
    int ed = edit_distance(cp_exp, cp_got);
    total_chars += cp_exp.size();
    total_char_edits += ed;
    double c_acc = cp_exp.empty() ? 1.0 : std::max(0.0, 1.0 - double(ed) / double(cp_exp.size()));

    ++total;
    if (rank == 1) ++top1_hits;
    if (rank >= 1 && rank <= 3) ++top3_hits;
    if (rank >= 1 && rank <= 5) ++top5_hits;
    if (rank >= 1) sum_mrr += 1.0 / double(rank);

    results.push_back({c, top1, rank, c_acc, top5});
  }

  api->destroy_session(sess);
  api->finalize();

  double top1_rate = total ? double(top1_hits) / total : 0.0;
  double top3_rate = total ? double(top3_hits) / total : 0.0;
  double mrr = total ? sum_mrr / total : 0.0;
  double char_acc = total_chars ? std::max(0.0, 1.0 - double(total_char_edits) / double(total_chars)) : 0.0;

  std::cout << "Cases=" << total
            << " | Top1=" << std::fixed << std::setprecision(2) << (top1_rate * 100.0) << "% (" << top1_hits << "/" << total << ")"
            << " | Top3=" << (top3_rate * 100.0) << "%"
            << " | MRR=" << std::setprecision(4) << mrr
            << " | CharAcc=" << std::setprecision(2) << (char_acc * 100.0) << "%\n";

  if (!out_json.empty()) {
    std::ofstream fo(out_json);
    fo << "{\n"
       << "  \"total\": " << total << ",\n"
       << "  \"top1_hits\": " << top1_hits << ",\n"
       << "  \"top3_hits\": " << top3_hits << ",\n"
       << "  \"top5_hits\": " << top5_hits << ",\n"
       << "  \"top1_acc\": " << std::setprecision(6) << top1_rate << ",\n"
       << "  \"top3_acc\": " << top3_rate << ",\n"
       << "  \"mrr\": " << mrr << ",\n"
       << "  \"char_acc\": " << char_acc << ",\n"
       << "  \"details\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
      const auto& r = results[i];
      fo << "    {\"category\": \"" << json_escape(r.c.category)
         << "\", \"mode\": \"" << json_escape(r.c.mode)
         << "\", \"pinyin\": \"" << json_escape(r.c.pinyin)
         << "\", \"expected\": \"" << json_escape(r.c.expected)
         << "\", \"top1\": \"" << json_escape(r.top1)
         << "\", \"rank\": " << r.found_rank
         << ", \"char_acc\": " << std::setprecision(4) << r.char_acc
         << ", \"top5\": [";
      for (size_t j = 0; j < r.top5.size(); ++j) {
        if (j) fo << ", ";
        fo << "\"" << json_escape(r.top5[j]) << "\"";
      }
      fo << "]}" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    fo << "  ]\n}\n";
  }

  return 0;
}
