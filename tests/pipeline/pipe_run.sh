# step 1: define dir
BASE_DIR=$(dirname "$(readlink -f "$0")")
BASELINE_DIR="$BASE_DIR/baseline"

#mkdir cache to store product and will be removed after test
mkdir -p "$BASE_DIR/pipe_cache"

GENERATE_LOG_DIR="$BASE_DIR/pipe_cache"
GENERATE_JSON_DIR="$BASE_DIR/pipe_cache"

# step 2: running scripts and execute `test_ci_pipeline.py` && running pytest
find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    if [ -d "$dir" ]; then
        find "$dir" -type f -name "*.sh" | while read -r file; do
            filename=$(basename "$file")
            extension="${filename##*.}"
            name="${filename%.$extension}"
            bash $file | tee "$GENERATE_LOG_DIR/$name.log"
            SCRIPT_EXITCODE=${PIPESTATUS[0]}
            if [ $SCRIPT_EXITCODE -ne 0 ]; then
                echo "Script has failed. Exit!"
                exit 1
            fi

            # begin to execute the logic of compare
            echo "$(dirname "$BASE_DIR")/st/st_utils/test_ci_st.py"
            pytest -x $(dirname "$BASE_DIR")/st/st_utils/test_ci_st.py \
                --baseline-json $BASELINE_DIR/$name.json \
                --generate-log $GENERATE_LOG_DIR/$name.log \
                --generate-json $GENERATE_JSON_DIR/$name.json
            PYTEST_EXITCODE=$?
            if [ $PYTEST_EXITCODE -ne 0 ]; then
                echo "${name} compare to baseline has failed, check it!"
                exit 1
            else
                echo "Pretrain ${name} execution success."
            fi
        done

        # python test testing
        find "$dir" -type f -name "*.py" | while read -r file; do
            pytest -x $file
        done
    fi
done

rm -rf "$BASE_DIR/pipe_cache"