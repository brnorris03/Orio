#include <stdio.h> 
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>

extern double getClock();

// -- begin caliper
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <string.h>

void print_help()
{
    const char* helpstr =
        "Usage: c-example [caliper-config(arg=...,),...]."
        "\nRuns \"runtime-report\" configuration by default."
        "\nUse \"none\" to run without a ConfigManager configuration."
        "\nAvailable configurations: ";

    puts(helpstr);
}
// -- end caliper

/*@ global @*/ 
/*@ external @*/

//int main(int argc, char * argv[]) {    // this is part of declarations
    /*@ declarations @*/

    /*@ prologue @*/

    // -- begin caliper
    cali_ConfigManager mgr;
    cali_ConfigManager_new(&mgr);

    const char* configstr = "none";

    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_help();
            return 0;
        } else {
            configstr = argv[1];
        }
    }

    if (strcmp(configstr, "none") == 0)
        configstr = "";

    //   Enable the requested performance measurement channels and start
    // profiling.
    cali_ConfigManager_add(&mgr, configstr);

    if (cali_ConfigManager_error(&mgr)) {
        cali_SHROUD_array errmsg;
        cali_ConfigManager_error_msg_bufferify(&mgr, &errmsg);
        fprintf(stderr, "Caliper config error: %s\n", errmsg.addr.ccharp);
        cali_SHROUD_memory_destructor(&errmsg.cxx);
    }
    cali_ConfigManager_start(&mgr);

    CALI_MARK_FUNCTION_BEGIN;
    // -- end caliper

    int orio_i;
    char coordstr[1024];
    sprintf(coordstr,"%s: %s",argv[0],"/*@ coordinate @*/");

    /*@ begin outer measurement @*/
    CALI_MARK_LOOP_BEGIN(reps_loop, "reps_loop");
    for (orio_i=0; orio_i<ORIO_REPS; orio_i++) {
        /*@ begin inner measurement @*/
        CALI_MARK_ITERATION_BEGIN(reps_loop,orio_i);

        /*@ tested code @*/

        CALI_MARK_ITERATION_END(reps_loop);
        /*@ end inner measurement @*/

        if(orio_i==0) {
            CALI_MARK_BEGIN("validation");
            /*@ validation code @*/
            CALI_MARK_END("validation");
        }
    }
    CALI_MARK_LOOP_END(reps_loop);
    /*@ end outer measurement @*/

    CALI_MARK_FUNCTION_END;

    //   Trigger output in all Caliper control channels.
    // This should be done after all measurement regions have been closed.
    cali_ConfigManager_flush(&mgr);
    cali_ConfigManager_delete(&mgr);

    /*@ epilogue @*/

    return 0;
}
