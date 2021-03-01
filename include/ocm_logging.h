#ifndef _OCM_LOGGING_
#define _OCM_LOGGING_

#include <iostream>
#include <string>
#include <sstream>

namespace ocm{

enum LoggingLevel {INFO,
                  WARNING,
                  ERROR,
                  FATAL
};


class Logger :  private std::streambuf , public std::ostream {
public: 
  Logger(const char* fname, int line, int log_level): std::ostream(this),fname_(fname),line_(line),severity_(log_level){
  std::cout<< "[" << fname_ <<"]"<<"\t"<<"["<<line<<"]"<<"\t";
  }
  static int MinLogLevel(){
    const char* ocm_log_env_var = std::getenv("OCM_LOG_LEVEL");
    if (ocm_log_env_var == nullptr) {
      return -1;
    }
    std::stringstream str_level;
    str_level << ocm_log_env_var;

    int int_level;
    if (!(str_level >> int_level)){
      // set to default level INFO, if incoorect ENV variable is declared
      int_level = 0;
    }
    return int_level;
  }

private:
  const char* fname_;
  int line_;
  int severity_;

  int overflow(int c) override{
    foo(c);
    return 0;
  }

  void foo(char c){
    std::cout.put(c);
  }

};

} // namespace ocm

#define OCM_LOG_ENABLED(level) (level <= ocm::Logger::MinLogLevel())
#define OCM_LOG(level) if (OCM_LOG_ENABLED(level))  ocm::Logger(__FILE__, __LINE__, level)

#endif // _OCM_LOGGING_