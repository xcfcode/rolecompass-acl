cp -r configs/* opencompass/configs/
cp -r datasets/* opencompass/opencompass/datasets/

# datasets
echo "from .socialbench import *  # noqa: F401, F403" >> "opencompass/opencompass/datasets/__init__.py"
echo "from .cross import *  # noqa: F401, F403" >> "opencompass/opencompass/datasets/__init__.py"
echo "from .hpd import *  # noqa: F401, F403" >> "opencompass/opencompass/datasets/__init__.py"
echo "from .incharacter import *  # noqa: F401, F403" >> "opencompass/opencompass/datasets/__init__.py"
echo "from .charactereval import *  # noqa: F401, F403" >> "opencompass/opencompass/datasets/__init__.py"

cp eval_rolecompass_.py opencompass/configs/