import unittest

from connectivity_conscious_mapper import _bfs
from connectivity_conscious_mapper import _find_root

class TestBFS(unittest.TestCase):
    def test_simple(self):
        # initialize the connectivity graph P
        P: dict[int, set[int]] = dict()
        
        # add vertices and edges in P
        P[0] = {1, 2, 3}
        P[1] = {0}
        P[2] = {0}
        P[3] = {0}

        # run BFS on the graph
        (length, path) = _bfs(1, P)

        # test that the path found is correct
        self.assertTrue(length == 2)

if __name__=='__main__':
	unittest.main()