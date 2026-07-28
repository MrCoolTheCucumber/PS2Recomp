#!/usr/bin/env bash

set -euo pipefail

usage()
{
    echo "usage: $0 --benchmark BIN --fixture DIR --pairs N --output DIR [options]" >&2
    echo "options: --cpu N --iterations N --warmup N --samples N" >&2
    echo "         --backend interpreter|recompiler|both" >&2
    echo "         --scope core|path1|all" >&2
}

benchmark=
fixture=
maximum_pairs=
output_directory=
cpu=
iterations=1000
warmup=10
samples=5
backend=both
scope=all

while (($# != 0)); do
    if (($# < 2)); then
        usage
        exit 2
    fi
    case "$1" in
    --benchmark)
        benchmark=$2
        ;;
    --fixture)
        fixture=$2
        ;;
    --pairs)
        maximum_pairs=$2
        ;;
    --output)
        output_directory=$2
        ;;
    --cpu)
        cpu=$2
        ;;
    --iterations)
        iterations=$2
        ;;
    --warmup)
        warmup=$2
        ;;
    --samples)
        samples=$2
        ;;
    --backend)
        backend=$2
        ;;
    --scope)
        scope=$2
        ;;
    *)
        usage
        exit 2
        ;;
    esac
    shift 2
done

if [[ -z "$benchmark" || -z "$fixture" ||
      -z "$maximum_pairs" || -z "$output_directory" ]]; then
    usage
    exit 2
fi
if [[ ! -x "$benchmark" ]]; then
    echo "benchmark is not executable: $benchmark" >&2
    exit 2
fi
if [[ ! -d "$fixture" ]]; then
    echo "fixture directory does not exist: $fixture" >&2
    exit 2
fi
if [[ -e "$output_directory" ]]; then
    echo "refusing to replace output directory: $output_directory" >&2
    exit 2
fi
case "$backend" in
interpreter)
    backends=(interpreter)
    ;;
recompiler)
    backends=(recompiler)
    ;;
both)
    backends=(interpreter recompiler)
    ;;
*)
    usage
    exit 2
    ;;
esac
case "$scope" in
core)
    scopes=(core)
    ;;
path1)
    scopes=(path1)
    ;;
all)
    scopes=(core path1)
    ;;
*)
    usage
    exit 2
    ;;
esac

for required_command in perf jq sha256sum mkfifo mktemp unlink; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "required command is unavailable: $required_command" >&2
        exit 2
    fi
done
if [[ -n "$cpu" ]] && ! command -v taskset >/dev/null 2>&1; then
    echo "taskset is required when --cpu is used" >&2
    exit 2
fi

mkdir -p "$output_directory"
temporary_directory=$(mktemp -d)
control_fifo="$temporary_directory/perf-control"
acknowledge_fifo="$temporary_directory/perf-ack"
cleanup()
{
    if [[ -e "$control_fifo" ]]; then
        unlink "$control_fifo"
    fi
    if [[ -e "$acknowledge_fifo" ]]; then
        unlink "$acknowledge_fifo"
    fi
    rmdir "$temporary_directory"
}
trap cleanup EXIT
mkfifo "$control_fifo" "$acknowledge_fifo"

benchmark_sha256=$(sha256sum "$benchmark" | awk '{print $1}')
perf_version=$(perf version)
primary_events=(
    cycles
    instructions
    branches
    branch-misses
    context-switches
)
cache_events=(
    L1-dcache-load-misses
    LLC-load-misses
    cache-misses
)
# The child shell, not this script, expands these positional parameters.
# shellcheck disable=SC2016
benchmark_wrapper='benchmark_stderr=$1; shift; exec "$@" 2>"$benchmark_stderr"'

summary_files=()
for selected_backend in "${backends[@]}"; do
    for selected_scope in "${scopes[@]}"; do
        prefix="$output_directory/$selected_backend-$selected_scope"
        primary_benchmark_output="$prefix.primary.jsonl"
        cache_benchmark_output="$prefix.cache.jsonl"
        primary_perf_output="$prefix.primary.perf.jsonl"
        cache_perf_output="$prefix.cache.perf.jsonl"
        summary_output="$prefix.summary.json"

        benchmark_command=(
            "$benchmark"
            "$fixture"
            "$maximum_pairs"
            --iterations "$iterations"
            --warmup "$warmup"
            --samples "$samples"
            --backend "$selected_backend"
            --scope "$selected_scope"
        )
        if [[ -n "$cpu" ]]; then
            benchmark_command=(
                taskset -c "$cpu"
                "${benchmark_command[@]}"
            )
        fi

        for counter_pass in primary cache; do
            if [[ "$counter_pass" == "primary" ]]; then
                pass_events=("${primary_events[@]}")
                benchmark_output="$primary_benchmark_output"
                perf_output="$primary_perf_output"
            else
                pass_events=("${cache_events[@]}")
                benchmark_output="$cache_benchmark_output"
                perf_output="$cache_perf_output"
            fi
            benchmark_stderr="$prefix.$counter_pass.stderr.log"
            perf_stderr="$prefix.$counter_pass.perf.stderr.log"
            event_list=$(IFS=,; echo "${pass_events[*]}")

            PS2X_PERF_CONTROL_FIFO="$control_fifo" \
            PS2X_PERF_ACK_FIFO="$acknowledge_fifo" \
            perf stat \
                --json-output \
                --no-csv-summary \
                --delay=-1 \
                --control="fifo:$control_fifo,$acknowledge_fifo" \
                --event "$event_list" \
                --output "$perf_output" \
                -- sh -c \
                    "$benchmark_wrapper" \
                    sh "$benchmark_stderr" "${benchmark_command[@]}" \
                >"$benchmark_output" \
                2>"$perf_stderr"
        done

        jq -n \
            --arg backend "$selected_backend" \
            --arg scope "$selected_scope" \
            --arg benchmark_sha256 "$benchmark_sha256" \
            --arg perf_version "$perf_version" \
            --arg cpu "${cpu:-unbound}" \
            --slurpfile primary_measurements "$primary_benchmark_output" \
            --slurpfile cache_measurements "$cache_benchmark_output" \
            --slurpfile primary_counters "$primary_perf_output" \
            --slurpfile cache_counters "$cache_perf_output" '
            def warm($measurements):
                [
                    $measurements[] |
                    select(.event == "warm-sample")
                ];
            def counter($counters; $prefix):
                ([
                    $counters[] |
                    select(.event | startswith($prefix))
                ][0] // null);
            (warm($primary_measurements)) as $primary_warm |
            (warm($cache_measurements)) as $cache_warm |
            (counter($primary_counters; "cycles")) as $cycles |
            {
                schema_version: 1,
                backend: $backend,
                timing_scope: (
                    $primary_warm[0].timing_scope // $scope
                ),
                benchmark_sha256: $benchmark_sha256,
                perf_version: $perf_version,
                cpu: $cpu,
                guest_pairs: (
                    [$primary_warm[].guest_pairs] | add // 0
                ),
                guest_cycles: (
                    [$primary_warm[].guest_cycles] | add // 0
                ),
                all_measurements_valid: (
                    all($primary_warm[]; .valid == true) and
                    all($cache_warm[]; .valid == true)
                ),
                counter_sets_complete: (
                    ($primary_counters | length) == 5 and
                    ($cache_counters | length) == 3
                ),
                all_supported_counters_full_time: (
                    all(
                        ($primary_counters +
                         $cache_counters)[];
                        if (
                            .["counter-value"] |
                            tostring |
                            startswith("<")
                        )
                        then true
                        else .["pcnt-running"] == 100
                        end
                    )
                ),
                unsupported_events: (
                    [
                        ($primary_counters +
                         $cache_counters)[] |
                        select(
                            .["counter-value"] |
                            tostring |
                            startswith("<")
                        ) |
                        .event
                    ]
                ),
                host_cycles: (
                    try ($cycles["counter-value"] | tonumber)
                    catch null
                ),
                host_cycles_per_guest_pair: (
                    if $cycles != null and
                       ([$primary_warm[].guest_pairs] | add // 0) != 0
                    then
                        try (
                            ($cycles["counter-value"] | tonumber) /
                            ([$primary_warm[].guest_pairs] | add)
                        ) catch null
                    else null
                    end
                ),
                counter_passes: [
                    {
                        name: "primary",
                        guest_pairs: (
                            [$primary_warm[].guest_pairs] | add // 0
                        ),
                        all_measurements_valid: (
                            all($primary_warm[]; .valid == true)
                        ),
                        counters: $primary_counters
                    },
                    {
                        name: "cache",
                        guest_pairs: (
                            [$cache_warm[].guest_pairs] | add // 0
                        ),
                        all_measurements_valid: (
                            all($cache_warm[]; .valid == true)
                        ),
                        counters: $cache_counters
                    }
                ],
                counters: ($primary_counters + $cache_counters)
            }
        ' >"$summary_output"
        summary_files+=("$summary_output")
    done
done

jq -s \
    --arg benchmark "$benchmark" \
    --arg fixture "$fixture" \
    --arg maximum_pairs "$maximum_pairs" \
    --arg iterations "$iterations" \
    --arg warmup "$warmup" \
    --arg samples "$samples" \
    '{
        schema_version: 1,
        benchmark: $benchmark,
        fixture: $fixture,
        maximum_pairs: ($maximum_pairs | tonumber),
        iterations: ($iterations | tonumber),
        warmup: ($warmup | tonumber),
        samples: ($samples | tonumber),
        runs: .
    }' \
    "${summary_files[@]}" \
    >"$output_directory/summary.json"

if ! jq -e '
    all(
        .runs[];
        .all_measurements_valid == true and
        .counter_sets_complete == true and
        .all_supported_counters_full_time == true and
        .guest_pairs > 0 and
        .counter_passes[0].guest_pairs ==
            .counter_passes[1].guest_pairs and
        .host_cycles != null
    )
' "$output_directory/summary.json" >/dev/null; then
    echo "invalid benchmark output or multiplexed perf counters" >&2
    exit 1
fi

cat "$output_directory/summary.json"
