import java.util.*;

public class ART1 {

    public static double choiceFunction(int[] I, double[] w, double alpha) {
        double intersection = 0;
        double wSum = 0;
        for (int i = 0; i < I.length; i++) {
            if (I[i] == 1 && w[i] == 1.0) {
                intersection++;
            }
            wSum += w[i];
        }
        return intersection / (alpha + wSum);
    }

    public static boolean vigilanceTest(int[] I, double[] w, double rho) {
        double intersection = 0;
        double iSum = 0;
        for (int i = 0; i < I.length; i++) {
            if (I[i] == 1 && w[i] == 1.0) {
                intersection++;
            }
            if (I[i] == 1) {
                iSum++;
            }
        }
        if (iSum == 0) return true;
        return (intersection / iSum) >= rho;
    }

    public static class Result {
        public List<double[]> prototypes;
        public List<Integer> assignments;

        public Result(List<double[]> prototypes, List<Integer> assignments) {
            this.prototypes = prototypes;
            this.assignments = assignments;
        }
    }

    public static Result train(List<int[]> patterns, double rho, double alpha, double beta, int maxEpochs) {
        int dim = patterns.get(0).length;
        List<double[]> prototypes = new ArrayList<>();
        int n = patterns.size();

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            boolean changed = false;
            for (int i = 0; i < n; i++) {
                int[] I = patterns.get(i);

                if (prototypes.isEmpty()) {
                    double[] newProto = new double[dim];
                    for (int d = 0; d < dim; d++) newProto[d] = I[d];
                    prototypes.add(newProto);
                    changed = true;
                    continue;
                }

                boolean[] disabled = new boolean[prototypes.size()];

                while (true) {
                    double bestT = -Double.MAX_VALUE;
                    int bestJ = -1;

                    for (int j = 0; j < prototypes.size(); j++) {
                        if (disabled[j]) continue;
                        double T = choiceFunction(I, prototypes.get(j), alpha);
                        if (T > bestT) {
                            bestT = T;
                            bestJ = j;
                        }
                    }

                    if (bestJ == -1 || bestT == -Double.MAX_VALUE) {
                        double[] newProto = new double[dim];
                        for (int d = 0; d < dim; d++) newProto[d] = I[d];
                        prototypes.add(newProto);
                        changed = true;
                        break;
                    }

                    if (vigilanceTest(I, prototypes.get(bestJ), rho)) {
                        double[] w = prototypes.get(bestJ);
                        double[] newW = new double[dim];
                        boolean protoChanged = false;
                        for (int d = 0; d < dim; d++) {
                            double andVal = (I[d] == 1 && w[d] == 1.0) ? 1.0 : 0.0;
                            newW[d] = beta * andVal + (1 - beta) * w[d];
                            if (newW[d] != w[d]) protoChanged = true;
                        }
                        if (protoChanged) {
                            prototypes.set(bestJ, newW);
                            changed = true;
                        }
                        break;
                    } else {
                        disabled[bestJ] = true;
                    }
                }
            }
            if (!changed) break;
        }

        // Final cluster assignments
        List<Integer> assignments = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int[] I = patterns.get(i);
            double bestT = -Double.MAX_VALUE;
            int bestJ = 0;
            for (int j = 0; j < prototypes.size(); j++) {
                double Tj = choiceFunction(I, prototypes.get(j), alpha);
                if (Tj > bestT) {
                    bestT = Tj;
                    bestJ = j;
                }
            }
            assignments.add(bestJ);
        }

        return new Result(prototypes, assignments);
    }

    public static void main(String[] args) {
        List<int[]> patterns = Arrays.asList(
            new int[]{1, 1, 1, 0, 0, 0},
            new int[]{1, 1, 0, 0, 0, 0},
            new int[]{0, 0, 0, 1, 1, 1},
            new int[]{0, 0, 0, 1, 1, 0},
            new int[]{1, 0, 1, 0, 0, 0}
        );

        double rho = 0.6;
        Result res = train(patterns, rho, 1e-6, 1.0, 10);

        System.out.println("Vigilance (rho) = " + rho + "\n");
        for (int j = 0; j < res.prototypes.size(); j++) {
            System.out.print("Category " + j + " Prototype: [");
            double[] p = res.prototypes.get(j);
            for (int d = 0; d < p.length; d++) {
                System.out.print((p[d] > 0.0 ? 1 : 0) + (d < p.length - 1 ? ", " : ""));
            }
            System.out.println("]");
        }

        System.out.println("\nPattern Assignments:");
        for (int i = 0; i < res.assignments.size(); i++) {
            System.out.println("Pattern " + i + " -> Category " + res.assignments.get(i));
        }
    }
}