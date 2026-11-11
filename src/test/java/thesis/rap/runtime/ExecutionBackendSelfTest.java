package thesis.rap.runtime;

import thesis.rap.alg.SiUniwardCostEngine;
import thesis.rap.app.StcPipelineIntegration;


public final class ExecutionBackendSelfTest {
    private ExecutionBackendSelfTest() {}

    public static void main(String[] args) {
        String previous = System.getProperty("rap.backend");
        try {
            System.setProperty("rap.backend", "CPU");
            assert ExecutionBackend.cpuForced();
            assert !ExecutionBackend.cudaPermitted();
            assert !ExecutionBackend.hipPermitted();
            assert SiUniwardCostEngine.configuredBackendName().equals("CPU");
            StcPipelineIntegration stc = new StcPipelineIntegration(10);
            assert stc.backendName().equals("CPU") : stc.backendName();
        } finally {
            if (previous == null) System.clearProperty("rap.backend");
            else System.setProperty("rap.backend", previous);
        }
        System.out.println("ExecutionBackendSelfTest: ALL TESTS PASSED");
    }
}
