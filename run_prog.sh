if [ "$#" !=  2 ]
    then
        echo "Usage: sh run_prog.sh <sample_input> <epochs_num(int)>
        - cifar10
        - cifar100
        - PetImages
        "
else
    for arg in "$@"
    do
        echo "=== Running Full Test using ${1} dataset ==="
        python -B "src\main.py" "${1}" "${2}"
    done
fi
