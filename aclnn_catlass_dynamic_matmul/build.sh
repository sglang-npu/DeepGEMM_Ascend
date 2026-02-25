#!/bin/bash
set -eo pipefail

# set params
# fixed params
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CATLASS_DIR="${SCRIPT_DIR}/catlass"
CATLASS_GIT_URL="https://gitcode.com/cann/catlass.git"
MSOPGEN_ROOT_DIR="${SCRIPT_DIR}/msopgen"
PYTHON_INCLUDE_DIR="/usr/local/python3.11.13/include/python3.11"
PYTHON_LIBRARY_DIR="/usr/local/python3.11.13/lib"

# variable params
DEVICE_VERSION=""
CLEAN_FLAG="false"
CUSTOM_CATLASS_FLAG="false"
CUSTOM_MSOPGEN_FLAG="false"

function download_catlass() {
    if [[ ! -d "${CATLASS_DIR}" ]]; then
        # create empty dir and clone catlass from git
        if ! git clone "${CATLASS_GIT_URL}" ; then
            echo "[DGA] [ERROR] get catlass code failed, please check error."
        fi
        CUSTOM_CATLASS_FLAG="true"
    else
        echo "[DGA] [INFO] catlass directory has already exist."
    fi
}

function msopgen_create() {
    if [[ -d "${MSOPGEN_ROOT_DIR}" ]]; then
        echo "[DGA] [INFO] msopgen directory has already exist."
        return
    fi
    CUSTOM_MSOPGEN_FLAG="true"
    msopgen gen -i ./catlass_dynamic_matmul.json \
        -c "ai_core-${DEVICE_VERSION}" \
        -lan cpp \
        -out "${MSOPGEN_ROOT_DIR}"
}

function copy_custom_code() {
    cp -r "${SCRIPT_DIR}"/op_host/* "${MSOPGEN_ROOT_DIR}"/op_host
    cp -r "${SCRIPT_DIR}"/op_kernel/* "${MSOPGEN_ROOT_DIR}"/op_kernel
}

function determine_device_type() {
    chip_name=$(npu-smi info -t board -i 0 -c 0 | awk '/Chip Name/ {print $NF}')
    npu_name=$(npu-smi info -t board -i 0 -c 0 | awk '/NPU Name/ {print $NF}')
    if [[ "$1" == "a2" ]]; then
        DEVICE_VERSION="Ascend${chip_name}"
    fi
    if [[ "$1" == "a3" ]]; then
        DEVICE_VERSION="${chip_name}_${npu_name}"
    fi
    echo "[DGA] [INFO] DEVICE_VERSION is ${DEVICE_VERSION}."
}

function parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --device-type)
                if [[ -z "$2" ]]; then
                    echo "[DGA] [ERROR] --device-type need value."
                    exit 1
                fi
                determine_device_type "$2"
                shift
                ;;
            --clean)
                CLEAN_FLAG="true"
                ;;
            *)
                echo "[DGA] [ERROR] wrong param $1."
                exit 1
            ;;
        esac
        shift
    done
}

function clean_cache() {
    if [[ ${CLEAN_FLAG} != "true" ]]; then
        echo "[DGA] [INFO] close clean flag."
        return
    fi
    # open clean flag will clean temp files
    if [[ -d "${CATLASS_DIR}" ]]; then
        echo "[DGA] [INFO] clean catlass directory."
        rm -rf "${CATLASS_DIR}"
    fi
    if [[ -d "${MSOPGEN_ROOT_DIR}" ]]; then
        echo "[DGA] [INFO] clean msopgen directory."
        rm -rf "${MSOPGEN_ROOT_DIR}"
    fi
}

function modify_host_cmake() {
    local HOST_CMAKE_FILE="${MSOPGEN_ROOT_DIR}/op_host/CMakeLists.txt"
    local ADD_CONTENT
    ADD_CONTENT=$(cat << EOF
aux_source_directory(\${CMAKE_CURRENT_SOURCE_DIR} ops_host_srcs)
aux_source_directory(\${CMAKE_CURRENT_SOURCE_DIR}/op_tiling ops_tiling_srcs)
set(ops_srcs \${ops_host_srcs} \${ops_tiling_srcs})
include_directories(
    \${ASCEND_HOME_PATH}/include
    \${ASCEND_HOME_PATH}/include/experiment/runtime
    \${ASCEND_HOME_PATH}/include/experiment/msprof
    \${ASCEND_HOME_PATH}/include/aclnn
    ${PYTHON_INCLUDE_DIR}
)
link_directories(\${ASCEND_HOME_PATH}/lib64 ${PYTHON_LIBRARY_DIR})
link_libraries(runtime python3.11)
EOF
)

    local TEMP_FILE="${MSOPGEN_ROOT_DIR}/op_host/CMakeLists.txt.tmp"
    touch "${TEMP_FILE}"
    echo "${ADD_CONTENT}" > "${TEMP_FILE}"
    sed '0,/^[[:space:]]*[^[:space:]]/ {/^[[:space:]]*[^[:space:]]/d}' "${HOST_CMAKE_FILE}" >> "${TEMP_FILE}"

    mv -f "${TEMP_FILE}" "${HOST_CMAKE_FILE}"
}

function modify_build_params() {
    if [[ ${CUSTOM_CATLASS_FLAG} == "true" ]]; then
        echo "[DGA] [INFO] custom some catlass code."
        sed -i 's/static size_t GetWorkspaceSize/CATLASS_HOST_DEVICE static size_t GetWorkspaceSize/' \
            "${CATLASS_DIR}/include/catlass/gemm/kernel/padding_matmul.hpp"
    fi
    # sed -i "s/customize/catlass/g" "${MSOPGEN_ROOT_DIR}/CMakePresets.json"

    if [[ ${CUSTOM_MSOPGEN_FLAG} == "true" ]]; then
        echo "[DGA] [INFO] custom some msopgen params."
        echo "add_ops_compile_options(ALL OPTIONS -I${CATLASS_DIR}/include)" \
            >> "${MSOPGEN_ROOT_DIR}/op_kernel/CMakeLists.txt"

        ESCAPED_PYTHON_INCLUDE_DIR=$(echo "$PYTHON_INCLUDE_DIR" | sed 's/[\/&\\]/\\&/g')
        ESCAPED_PYTHON_LIBRARY_DIR=$(echo "$PYTHON_LIBRARY_DIR" | sed 's/[\/&\\]/\\&/g')
        sed -i "s/-lexe_graph/-I ${ESCAPED_PYTHON_INCLUDE_DIR} -lexe_graph/" "${MSOPGEN_ROOT_DIR}/cmake/func.cmake"
        sed -i "s/-lexe_graph/-L ${ESCAPED_PYTHON_LIBRARY_DIR} -lexe_graph/" "${MSOPGEN_ROOT_DIR}/cmake/func.cmake"
        sed -i "s/-lexe_graph/-lruntime -lpython3.11 -lexe_graph/" "${MSOPGEN_ROOT_DIR}/cmake/func.cmake"
        modify_host_cmake
    fi
}

function build_package() {
    export LD_LIBRARY_PATH=${PYTHON_LIBRARY_DIR}:${LD_LIBRARY_PATH}
    cd "${MSOPGEN_ROOT_DIR}"
    bash build.sh
    cd -
}

function main() {
    parse_args "$@"
    clean_cache
    download_catlass
    msopgen_create
    copy_custom_code
    modify_build_params
    build_package
}

main "$@"