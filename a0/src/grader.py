#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect, collections, os, pickle, gzip
from graderUtil import graded, CourseTestRunner, GradedTestCase
from collections import defaultdict

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################
NO_POINTS_STATEMENT = "OPTIONAL ASSIGNMENT.\nTHIS DOES NOT CONTRIBUTE TO YOUR FINAL GRADE IN ANY WAY.\n(i.e., the assignment where everything's made up and the points don't matter.)\n\n"

def randvec():
  v = defaultdict(float)
  for _ in range(10):
    v[random.randint(0, 10)] = random.randint(0, 10) - 5
  return v

def genSentence(K, L):
  # K = alphabet size, L = length
  return ' '.join(str(random.randint(0, K)) for _ in range(L))

def genLimitedCharacterString(numChars, length):
    # Generate a random string of the given length
    return ' '.join(chr(random.randint(ord('a'), ord('a') + numChars - 1)) for _ in range(length))

#########
# TESTS #
#########
class Test_3a(GradedTestCase):

  @graded(student_feedback=NO_POINTS_STATEMENT)
  def test_0(self):
    """3a-0-basic:  Simple test case."""
    self.assertEqual('word',
                     submission.findAlphabeticallyLastWord('which is the last word alphabetically'))
  
  @graded(student_feedback=NO_POINTS_STATEMENT)
  def test_1(self):
    """3a-1-basic:  Simple test case."""
    self.assertEqual('sun',
                     submission.findAlphabeticallyLastWord('cat sun dog')) 
  
  @graded(student_feedback=NO_POINTS_STATEMENT)
  def test_2(self):
    """3a-2-basic:  Big test case."""
    self.assertEqual('99999',
                     submission.findAlphabeticallyLastWord(' '.join(str(x) for x in range(100000))))
class Test_3b(GradedTestCase):
  @graded(student_feedback=NO_POINTS_STATEMENT)
  def test_0(self):
    """3b-0-basic:  Simple test case."""
    self.assertEqual(5,
                     submission.euclideanDistance((1, 5), (4, 1)))
  @graded(is_hidden=True, student_feedback=NO_POINTS_STATEMENT)
  def test_1(self):
    """3b-1-hidden:  100 random trials."""
    random.seed(42)
    for _ in range(100):
      x1 = random.randint(0, 10)
      y1 = random.randint(0, 10)
      x2 = random.randint(0, 10)
      y2 = random.randint(0, 10)
      def comp(test_func):
        return test_func((x1, y1), (x2, y2))
      self.compare_with_solution_or_wait(submission, 'euclideanDistance', comp)
class Test_3c(GradedTestCase):
  @graded(student_feedback=NO_POINTS_STATEMENT)
  def test_0(self):
    """3c-0-basic:  Simple test case."""
    self.assertEqual(sorted(['a a a a a']),
                     sorted(submission.mutateSentences('a a a a a')))
    self.assertEqual(sorted(['the cat']),
                     sorted(submission.mutateSentences('the cat')))
    self.assertEqual(sorted(['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']),
                     sorted(submission.mutateSentences('the cat and the mouse')))

  @graded(is_hidden=True, student_feedback=NO_POINTS_STATEMENT)
  def test_1(self):
    """3c-1-hidden:  Small random trials."""
    random.seed(42)
    for _ in range(10):
      sentence = genSentence(3, 5)
      solution_mutateSentences = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.mutateSentences(sentence))
      self.assertEqual(sorted(solution_mutateSentences),
                       sorted(submission.mutateSentences(sentence)))

  @graded(is_hidden=True, student_feedback=NO_POINTS_STATEMENT)
  def test_2(self):
    """3c-2-hidden:  Large random trials."""
    random.seed(42)
    for _ in range(10):
      sentence = genSentence(25, 10)
      solution_mutateSentences = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.mutateSentences(sentence))
      self.assertEqual(sorted(solution_mutateSentences),
                       sorted(submission.mutateSentences(sentence)))
class Test_3d(GradedTestCase):

  @graded(student_feedback=NO_POINTS_STATEMENT)
  def test_0(self):
    """3d-0-basic:  Simple test case."""
    self.assertEqual(15,
                     submission.sparseVectorDotProduct(defaultdict(float, {'a': 5}), defaultdict(float, {'b': 2, 'a': 3})))

  @graded(is_hidden=True, student_feedback=NO_POINTS_STATEMENT)
  def test_1(self):
    """3d-1-basic:  Random trials."""
    random.seed(42)
    for _ in range(10):
      v1 = randvec()
      v2 = randvec()
      def comp(test_func):
        return test_func(v1, v2)
      self.compare_with_solution_or_wait(submission, 'sparseVectorDotProduct', comp)
class Test_3e(GradedTestCase):

  @graded(student_feedback=NO_POINTS_STATEMENT)
  def test_0(self):
    """3e-0-basic:  Simple test case."""
    v = defaultdict(float, {'a': 5})
    submission.incrementSparseVector(v, 2, defaultdict(float, {'b': 2, 'a': 3}))
    self.assertEqual(defaultdict(float, {'a': 11, 'b': 4}), v, msg='Simple test.')

  @graded(is_hidden=True, student_feedback=NO_POINTS_STATEMENT)
  def test_1(self):
    """3e-1-hidden:  Random trials."""
    random.seed(42)
    for _ in range(10):
      v1a = randvec()
      v1b = v1a.copy()
      v2 = randvec()
      def comp(func, v):
        func(v, 4, v2)
        for key in list(v):
          if v[key] == 0:
            del v[key]
        return v
      ans1 = comp(submission.incrementSparseVector, v1a)
      ans2 = self.run_with_solution_if_possible(submission, lambda sub_or_sol: comp(sub_or_sol.incrementSparseVector, v1b))
      for key in ans2.keys():
        self.assertEqual(ans1, ans2)

class Test_3f(GradedTestCase):

  @graded(student_feedback=NO_POINTS_STATEMENT)
  def test_0(self):
    """3f-0-basic:  Simple test case."""
    self.assertEqual(set(['quick', 'brown', 'jumps', 'over', 'lazy']),
                     submission.findSingletonWords('the quick brown fox jumps over the lazy fox'))

  @graded(is_hidden=True, student_feedback=NO_POINTS_STATEMENT)
  def test_1(self):
    """3f-1-basic:  Random trials."""
    numTokens, numTypes = (1000, 10)
    import random
    random.seed(42)
    text = ' '.join(str(random.randint(0, numTypes)) for _ in range(numTokens))
    def comp(test_func):
      return test_func(text)
    self.compare_with_solution_or_wait(submission, 'findSingletonWords', comp)

  @graded(is_hidden=True, student_feedback=NO_POINTS_STATEMENT)
  def test_2(self):
    """3f-2-hidden:  Random trials (bigger)."""
    numTokens, numTypes = (10000, 100)
    import random
    random.seed(42)
    text = ' '.join(str(random.randint(0, numTypes)) for _ in range(numTokens))
    def comp(test_func):
      return test_func(text)
    self.compare_with_solution_or_wait(submission, 'findSingletonWords', comp)
class Test_3g(GradedTestCase):

  @graded(student_feedback=NO_POINTS_STATEMENT)
  def test_0(self):
    """3g-0-basic:  Simple test cases."""
    self.assertEqual(0,
                     submission.computeLongestPalindromeLength(""),
                     msg='simple test')
    self.assertEqual(1,
                     submission.computeLongestPalindromeLength("a"),
                     msg='simple test')
    self.assertEqual(2,
                     submission.computeLongestPalindromeLength("aa"),
                     msg='simple test')
    self.assertEqual(1,
                     submission.computeLongestPalindromeLength("ab"),
                     msg='simple test')
    self.assertEqual(3,
                     submission.computeLongestPalindromeLength("animal"),
                     msg='simple test')

  @graded(is_hidden=True, timeout=1, student_feedback=NO_POINTS_STATEMENT)
  def test_1(self):
    """3g-1-hidden:  Random trials."""
    import random
    random.seed(42)
    # Generate a random string of the given length
    text = genLimitedCharacterString(2,10)
    def comp(test_func):
      return test_func(text)
    self.compare_with_solution_or_wait(submission, 'computeLongestPalindromeLength', comp)

  @graded(is_hidden=True, timeout=1, student_feedback=NO_POINTS_STATEMENT)
  def test_2(self):
    """3g-2-hidden:  Random trials (more characters)"""
    import random
    random.seed(42)
    # Generate a random string of the given length
    text = genLimitedCharacterString(10,10)
    def comp(test_func):
      return test_func(text)
    self.compare_with_solution_or_wait(submission, 'computeLongestPalindromeLength', comp)

  @graded(is_hidden=True, timeout=1, student_feedback=NO_POINTS_STATEMENT)
  def test_3(self):
    """3g-3-hidden:  Random trials (long)."""
    import random
    random.seed(42)
    # Generate a random string of the given length
    text = genLimitedCharacterString(5,20)
    def comp(test_func):
      return test_func(text)
    self.compare_with_solution_or_wait(submission, 'computeLongestPalindromeLength', comp)

  @graded(is_hidden=True, timeout=2, student_feedback=NO_POINTS_STATEMENT)
  def test_4(self):
    """3g-4-hidden:  Random trials (longer)."""
    import random
    random.seed(42)
    # Generate a random string of the given length
    text = genLimitedCharacterString(5,400)
    def comp(test_func):
      return test_func(text)
    self.compare_with_solution_or_wait(submission, 'computeLongestPalindromeLength', comp)

def getTestCaseForTestID(test_id):
  question, part, _ = test_id.split('-')
  g = globals().copy()
  for name, obj in g.items():
    if inspect.isclass(obj) and name == ('Test_'+question):
      return obj('test_'+part)

if __name__ == '__main__':
  # Parse for a specific test
  parser = argparse.ArgumentParser()
  parser.add_argument('test_case', nargs='?', default='all')
  test_id = parser.parse_args().test_case

  assignment = unittest.TestSuite()
  if test_id != 'all':
    assignment.addTest(getTestCaseForTestID(test_id))
  else:
    assignment.addTests(unittest.defaultTestLoader.discover('.', pattern='grader.py'))
  CourseTestRunner().run(assignment)
