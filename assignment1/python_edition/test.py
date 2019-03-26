from benchmark import *
from optimise import *

def test_for_first_submission():
    # arithmetic_recombination = arithmetic_recombination_factory(.3)
    numRuns = 50
    for benchmark, budget in zip(benchmark_lst, evaluations):
        for s, select in enumerate(selection):
            for r, recombine in enumerate(recombination):
                for m, mutate in enumerate(mutation):
                    gen_record = [sys.float_info.max]
                    best = Solution(None, None, 0)
                    for i in range(numRuns):
                        print("=====================================================================")
                        print("[Optimisation of function %s, crossoverIdx=%d, mutationId=%d, selectionIdx=%d]" %(benchmark[0].__name__, r, m, s))
                        record, p = optimise(benchmark, budget, select, recombine, mutate)
                        print("RUN %d: Approximate optimal value=%.16f" % (i, record[-1]))
                        print("RUN %d: Approximate optimal optimum=%s" % (i, str(p.vec)))
                        if record[-1] < gen_record[-1]:
                            best = p
                        gen_record.append(record[-1])
                    print("=====================================================================")
                    out_record = np.array(gen_record[1:])
                    print("Testing " + benchmark[0].__name__)
                    print("Best value is: " , out_record.min())
                    print("Best solution is: " + str(best.vec))
                    print("Average value is: " , out_record.mean())
                    print("Var is " , out_record.var())

if __name__ == "__main__":
    test_for_first_submission()