import unittest
import pandas as pd
from pathlib import Path
import sys
import os
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.utils.gcs_utils import set_environement_variable
is_env_variables_set = set_environement_variable()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/gauthies/sdd/sdd-general/dist/auth.json'

from src.utils.utils import get_top_k_classification_report
classes = ['rosacea_inflammatory', 'atopic_dermatitis', 'rosacea_erythemato_telangiectasique','peri_oral_dermatitis',
                'seborrheic_keratosis','psoriasis_vulgar','seborrheic_dermatitis','nummular_eczema',
                'tinea_versicolor','chronic_hand_eczema','vulgar_warts','folliculitis','alopecia_androgenic',
                'dyshidrosis','nevus','melasma','alopecia_areata','intertrigo','urticaria','vitiligo','keratosis_pilaris',
                'molluscum','cheilitis_eczematous','tinea_corporis','prurigo_nodularis','actinic_keratosis',
                'genital_warts','plane_warts','pityriasis_rosae','melanonychia','psoriasis_pustular_palmoplantar',
                'granuloma_annulare','psoriasis_guttate','lichen_simplex_chronicus','shingles','herpes_simplex',
                'acne_cystic', 'acne_scars', 'acne_excoriated', 'acne_comedos', 'acne_mixed',]

class UntilsTests(unittest.TestCase):
    def test_get_top_k_classification_report(self):
        k=3
        df_preds = pd.read_csv('gs://oro-ds-test-bucket/sdd_acne_files/unittests/utils_test_predictions.csv')
        report = get_top_k_classification_report(df_preds, k=k, classes=classes)
        
        prob_columns = ['prob_' + disease for disease in classes]
        for _, row in df_preds.iterrows():
            for i in range(k):
                # 1. Test if Pred1, Pred2, Pred3 are accurate
                self.assertEqual(row[prob_columns].sort_values(ascending=False).index[i].replace('prob_', ''), row[f'Pred{i+1}'])
                # 2. Test if the top-3 accuracy is accurate
                label_in_top_k = len(set([row['label']]).intersection(set(row[['Pred1', 'Pred2', 'Pred3']]))) == 1
                self.assertEqual( label_in_top_k ,row['top3_prediction'] )
   
        # 3. Test if the top-3 accuracy value is computed correctly
        accuracy = report['support']['accuracy']
        file_accuracy = df_preds[df_preds['top3_prediction']==True].shape[0] / df_preds.shape[0]
        self.assertEqual(accuracy,  file_accuracy)
                


if __name__ == '__main__':
    unittest.main()