# export MSOT_ROOT=
# export MSOT_TESTING=

# export ROBOT_ROOT=
# export ROBOT_DEPENDENCIES_ROOT=

FAIL=0

if [ -n "$BASH_VERSION" ]; then
    function check_env_var {
        if [ -z "${!1}" ]; then
            echo "[ERR] $1 is not set"
            FAIL=1
        fi
    }
elif [ -n "$ZSH_VERSION" ]; then
    function check_env_var {
        val=$(echo ${${(Mk)parameters:#$1}/(#m)*/${(P)MATCH}})
        if [ -z "$val" ]; then
            echo "[ERR] $1 is not set"
            FAIL=1
        fi
    }
else
    echo '[ERR] unknown shell'
    return 1
fi

check_env_var MSOT_ROOT
check_env_var ROBOT_ROOT
check_env_var ROBOT_DEPENDENCIES_ROOT

if [ $FAIL -eq 1 ]; then
    echo '[FATAL] missing environment variables, aborting'
    return 1
fi

export PYTHONPATH=$PYTHONPATH:$ROBOT_ROOT:$MSOT_ROOT:$ROBOT_DEPENDENCIES_ROOT
