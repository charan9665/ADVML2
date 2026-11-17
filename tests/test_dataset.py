import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))
from data_loader import DATA, load_dataset

@unittest.skipUnless(all((DATA/name).exists() for name in ("movies.dat","ratings.dat","users.dat")),"Install official MovieLens 1M files to run data checks")
class MovieLensDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.movies,cls.ratings,cls.users=load_dataset()

    def test_official_dataset_dimensions(self):
        self.assertEqual(len(self.movies),3883)
        self.assertEqual(len(self.ratings),1000209)
        self.assertEqual(len(self.users),6040)

    def test_missing_duplicates_genres_and_timestamps(self):
        for table in self.movies.attrs["inspection"].values():
            self.assertEqual(table["missing_cells"],0)
            self.assertEqual(table["duplicate_rows"],0)
        self.assertTrue(self.movies.GenreList.map(bool).any())
        self.assertTrue(str(self.ratings.TimestampDate.dtype).startswith("datetime64[ns, UTC]"))

    def test_rating_distribution(self):
        self.assertEqual(self.ratings.Rating.value_counts().sort_index().to_dict(),{1:56174,2:107557,3:261197,4:348971,5:226310})

if __name__=="__main__":unittest.main()
