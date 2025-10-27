# clean all temporary theory files in a given directory

ROOT_DIR=$1

if [ -z "$ROOT_DIR" ]; then
    echo "Usage: $0 <root_dir>"
    exit 1
fi

find $ROOT_DIR -name "Extract_*.thy" -exec rm -f {} \;
find $ROOT_DIR -name "Transitions_*.thy" -exec rm -f {} \;
find $ROOT_DIR -name "Rediscover_*.thy" -exec rm -f {} \;
find $ROOT_DIR -name "TemplateData_*.thy" -exec rm -f {} \;
find $ROOT_DIR -name "Eval_Template_*.thy" -exec rm -f {} \;