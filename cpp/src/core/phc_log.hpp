#pragma once

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace phc::log {

enum class Level { Error, Warn, Info, Debug };

inline Level ParseLevel(const char* s) {
  if (!s || !*s) {
    return Level::Info;
  }
  auto eq = [&](const char* lit) {
    return std::strlen(lit) == std::strlen(s) &&
           std::strncmp(s, lit, std::strlen(lit)) == 0;
  };
  if (eq("error")) {
    return Level::Error;
  }
  if (eq("warn") || eq("warning")) {
    return Level::Warn;
  }
  if (eq("info")) {
    return Level::Info;
  }
  if (eq("debug")) {
    return Level::Debug;
  }
  return Level::Info;
}

inline Level CurrentLevel() {
  static const Level kLevel = ParseLevel(std::getenv("PHC_LOG_LEVEL"));
  return kLevel;
}

inline bool IsEnabled(Level msg) {
  return static_cast<int>(msg) <= static_cast<int>(CurrentLevel());
}

inline const char* Tag(Level lvl) {
  switch (lvl) {
    case Level::Error:
      return "ERROR";
    case Level::Warn:
      return "WARN";
    case Level::Info:
      return "INFO";
    case Level::Debug:
      return "DEBUG";
  }
  return "?";
}

// Reduce libcamera's own INFO spam unless the operator set levels explicitly.
inline void ConfigureThirdPartyLogLevels() {
  if (!std::getenv("LIBCAMERA_LOG_LEVELS")) {
    ::setenv("LIBCAMERA_LOG_LEVELS", "*:ERROR", 0);
  }
}

class LogStream {
 public:
  explicit LogStream(Level lvl) : lvl_(lvl), enabled_(IsEnabled(lvl)) {}
  ~LogStream() {
    if (enabled_ && wrote_) {
      std::cerr << '\n';
    }
  }

  LogStream(const LogStream&) = delete;
  LogStream& operator=(const LogStream&) = delete;

  template <typename T>
  LogStream& operator<<(const T& value) {
    if (!enabled_) {
      return *this;
    }
    if (!wrote_) {
      std::cerr << '[' << Tag(lvl_) << "] ";
      wrote_ = true;
    }
    std::cerr << value;
    return *this;
  }

 private:
  Level lvl_;
  bool enabled_;
  bool wrote_ = false;
};

inline LogStream Error() { return LogStream(Level::Error); }
inline LogStream Warn() { return LogStream(Level::Warn); }
inline LogStream Info() { return LogStream(Level::Info); }
inline LogStream Debug() { return LogStream(Level::Debug); }

}  // namespace phc::log
