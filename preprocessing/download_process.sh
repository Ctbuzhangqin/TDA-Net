#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${TDA_PROCESSED_DIR:-${SCRIPT_DIR}/../dataset/processed_data}"
DOWNLOAD_DIR="${OUTPUT_DIR}/downloads"
TMP_DIR="${OUTPUT_DIR}/tmp"
GRID_FILE="${SCRIPT_DIR}/grid"
ZAXIS_FILE="${SCRIPT_DIR}/zaxis.txt"
BASE_URL="https://downloads.psl.noaa.gov/Datasets/godas"

YEARS=("$@")
if [[ ${#YEARS[@]} -eq 0 ]]; then
    echo "Usage: bash preprocessing/download_process.sh YEAR [YEAR ...]" >&2
    exit 2
fi
VARS_2D=(sshg uflx vflx)
VARS_3D=(salt pottmp ucur vcur)
TARGET_LEVELS="10,15,30,50,75,100,125,150,200,250,300,400,500,600,800,1000"


check_dependencies() {
    command -v wget >/dev/null 2>&1 || { echo "wget not found"; exit 1; }
    command -v cdo >/dev/null 2>&1 || { echo "cdo not found"; exit 1; }
    [[ -f "${GRID_FILE}" ]] || { echo "Missing grid file: ${GRID_FILE}"; exit 1; }
    [[ -f "${ZAXIS_FILE}" ]] || { echo "Missing zaxis file: ${ZAXIS_FILE}"; exit 1; }
}

prepare_directories() {
    mkdir -p "${OUTPUT_DIR}" "${DOWNLOAD_DIR}" "${TMP_DIR}"
}

download_file() {
    local var="$1"
    local year="$2"
    local raw_file="${DOWNLOAD_DIR}/${var}-${year}.nc"
    local url="${BASE_URL}/${var}.${year}.nc"

    if [[ ! -s "${raw_file}" ]]; then
        echo "Downloading ${var} ${year} -> ${raw_file}"
        wget "${url}" -O "${raw_file}"
    else
        echo "Download exists, skip: ${raw_file}"
    fi
}

process_2d_var() {
    local var="$1"
    local year="$2"
    local raw_file="${DOWNLOAD_DIR}/${var}-${year}.nc"
    local processed_file="${OUTPUT_DIR}/${var}-${year}-processed.nc"

    if [[ ! -s "${processed_file}" ]]; then
        echo "Processing 2D ${var} ${year} -> ${processed_file}"
        cdo -b f64 remapbil,"${GRID_FILE}" "${raw_file}" "${processed_file}"
    else
        echo "Processed file exists, skip: ${processed_file}"
    fi
}

process_3d_var() {
    local var="$1"
    local year="$2"
    local raw_file="${DOWNLOAD_DIR}/${var}-${year}.nc"
    local remap_file="${TMP_DIR}/${var}-${year}-remap.nc"
    local intlevel_file="${TMP_DIR}/${var}-${year}-intlevel.nc"
    local processed_file="${OUTPUT_DIR}/${var}-${year}-processed.nc"

    if [[ ! -s "${remap_file}" ]]; then
        echo "Remapping 3D ${var} ${year} -> ${remap_file}"
        cdo -b f64 remapbil,"${GRID_FILE}" "${raw_file}" "${remap_file}"
    else
        echo "Remap file exists, skip: ${remap_file}"
    fi

    if [[ "${var}" == "pottmp" ]]; then
        local sst_file="${OUTPUT_DIR}/sst-${year}-processed.nc"
        if [[ ! -s "${sst_file}" ]]; then
            echo "Extracting sst from ${var} ${year} level=5 -> ${sst_file}"
            cdo chname,pottmp,sst -vertmean -sellevel,5 "${remap_file}" "${sst_file}"
        else
            echo "SST file exists, skip: ${sst_file}"
        fi
    fi

    if [[ ! -s "${intlevel_file}" ]]; then
        echo "Interpolating levels for ${var} ${year} -> ${intlevel_file}"
        cdo intlevel,"${TARGET_LEVELS}" "${remap_file}" "${intlevel_file}"
    else
        echo "Intlevel file exists, skip: ${intlevel_file}"
    fi

    if [[ ! -s "${processed_file}" ]]; then
        echo "Setting z-axis for ${var} ${year} -> ${processed_file}"
        cdo setzaxis,"${ZAXIS_FILE}" "${intlevel_file}" "${processed_file}"
    else
        echo "Processed file exists, skip: ${processed_file}"
    fi
}

main() {
    check_dependencies
    prepare_directories

    for year in "${YEARS[@]}"; do
        for var in "${VARS_2D[@]}"; do
            download_file "${var}" "${year}"
            process_2d_var "${var}" "${year}"
        done

        for var in "${VARS_3D[@]}"; do
            download_file "${var}" "${year}"
            process_3d_var "${var}" "${year}"
        done
    done
}

main "$@"
