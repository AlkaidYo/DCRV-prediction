import sys
sys.path.append('../')
from config.file_path_config import xgb_cached_model_path, base_path
import logging
import warnings
import joblib
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(funcName)s %(lineno)d %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main_start():
    model = joblib.load(xgb_cached_model_path)
    importance = model.get_score(importance_type='weight')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    sorted_features = ['Cell in Row(7d)', 'Page(7d)', 'Cell(7d) ', 'Burst Diff Delta(7d)', 'CE in Row(7d)',
                       'Row Max Multi Bit(28d)', 'CE in Column(28d)', 'Data Pin(7d)',
                       'Hard Error Cell(28d)', 'Column(28d)',
                       'CE Max(1d)', 'CE Std(7d)', 'Bank Dist(7d)',
                       'Row in Span Region(28d)', 'CE in Parity Device(7d)']
    sorted_scores = [x[1] for x in sorted_importance][:15]


    plt.figure(figsize=(9, 5))
    plt.barh(sorted_features, sorted_scores, color='#666666', height=0.65)
    plt.xlabel('Feature Importance Score')
    plt.gca().invert_yaxis()
    fig = plt.gcf()
    fig.subplots_adjust(left=0.3)
    plt.savefig(base_path + 'figure/figure_10.pdf', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main_start()
