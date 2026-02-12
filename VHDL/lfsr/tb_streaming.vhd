--------------------------------------------------------------------------------
-- Testbench: tb_streaming
--
-- Simple streaming testbench for lfsr/fulltopHDC:
--   1. Generate NUM_SAMPLES frames of FEATURE_SIZE pixels
--   2. Drive AXI-Stream-like inputs (TVALID_M/TDATA_M/TLAST_M)
--   3. Wait for classification output (TVALID_S/TDATA_S)
--
-- No learning or AXI-Lite control is used in this variant.
--------------------------------------------------------------------------------

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
USE IEEE.MATH_REAL.ALL;

ENTITY tb_streaming IS
END ENTITY tb_streaming;

ARCHITECTURE sim OF tb_streaming IS

    -- Constants matching lfsr/fulltopHDC generics
    CONSTANT CLK_PERIOD   : TIME    := 20 ns;  -- 50 MHz clock
    CONSTANT INBIT        : INTEGER := 8;
    CONSTANT DIMENSION    : INTEGER := 1000;
    CONSTANT LOGFEATURE   : INTEGER := 10;
    CONSTANT CLASSES      : INTEGER := 10;
    CONSTANT FEATURE_SIZE : INTEGER := 784;
    CONSTANT CLASS_MEM    : INTEGER := 7;
    CONSTANT CONF_COMP    : INTEGER := 3;
    CONSTANT RSA_ZPAD     : INTEGER := 1;
    CONSTANT COMP_ZPAD    : INTEGER := 6;
    CONSTANT LOG_CLASSES  : INTEGER := 4;
    CONSTANT LOGN         : INTEGER := 2;
    CONSTANT ID_REMAINDER : INTEGER := 232;
    CONSTANT ID_COEFF     : INTEGER := 3;

    -- Number of samples to simulate
    CONSTANT NUM_SAMPLES : INTEGER := 20;

    -- DUT signals
    SIGNAL clk      : STD_LOGIC := '0';
    SIGNAL rst      : STD_LOGIC := '0';

    -- AXI-Stream Master interface (source -> HDC)
    SIGNAL TVALID_M : STD_LOGIC := '0';
    SIGNAL TDATA_M  : STD_LOGIC_VECTOR(INBIT - 1 DOWNTO 0) := (OTHERS => '0');
    SIGNAL TKEEP_M  : STD_LOGIC_VECTOR(0 DOWNTO 0) := "1";
    SIGNAL TREADY_M : STD_LOGIC;
    SIGNAL TLAST_M  : STD_LOGIC := '0';

    -- AXI-Stream Slave interface (HDC -> sink)
    SIGNAL TVALID_S : STD_LOGIC;
    SIGNAL TDATA_S  : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL TKEEP_S  : STD_LOGIC_VECTOR(0 DOWNTO 0);
    SIGNAL TREADY_S : STD_LOGIC := '1';
    SIGNAL TLAST_S  : STD_LOGIC;

    -- Simulation control
    SIGNAL sim_done : BOOLEAN := FALSE;

    -- For pseudo-random data generation
    SHARED VARIABLE seed1 : POSITIVE := 1;
    SHARED VARIABLE seed2 : POSITIVE := 2;

BEGIN

    -- Clock generation
    clk_process : PROCESS
    BEGIN
        WHILE NOT sim_done LOOP
            clk <= '0';
            WAIT FOR CLK_PERIOD / 2;
            clk <= '1';
            WAIT FOR CLK_PERIOD / 2;
        END LOOP;
        WAIT;
    END PROCESS;

    -- Device Under Test instantiation
    DUT : ENTITY work.fulltopHDC
        GENERIC MAP (
            inbit                 => INBIT,
            dimension             => DIMENSION,
            pruning               => 336,
            logfeature            => LOGFEATURE,
            classes               => CLASSES,
            featureSize           => FEATURE_SIZE,
            classMemSize          => CLASS_MEM,
            confCompNum           => CONF_COMP,
            rsaZeropadding        => RSA_ZPAD,
            comparatorZeroPadding => COMP_ZPAD,
            logClasses            => LOG_CLASSES,
            logn                  => LOGN,
            IDreminder            => ID_REMAINDER,
            IDcoefficient         => ID_COEFF,
            lenTKEEP_M            => 1,
            lenTDATA_S            => 8,
            lenTKEEP_S            => 1
        )
        PORT MAP (
            clk      => clk,
            rst      => rst,
            TVALID_M => TVALID_M,
            TDATA_M  => TDATA_M,
            TKEEP_M  => TKEEP_M,
            TREADY_S => TREADY_S,
            TLAST_M  => TLAST_M,
            TREADY_M => TREADY_M,
            TVALID_S => TVALID_S,
            TLAST_S  => TLAST_S,
            TDATA_S  => TDATA_S,
            TKEEP_S  => TKEEP_S
        );

    ----------------------------------------------------------------------------
    -- Stimulus process
    ----------------------------------------------------------------------------
    stimulus : PROCESS
        VARIABLE rand_val   : REAL;
        VARIABLE pixel_val  : INTEGER;
        VARIABLE sample_idx : INTEGER;
    BEGIN
        -- Reset sequence (active-low)
        rst <= '0';
        WAIT FOR CLK_PERIOD * 10;
        rst <= '1';
        WAIT FOR CLK_PERIOD * 10;

        REPORT "=== Starting LFSR Streaming Simulation ===" SEVERITY NOTE;
        REPORT "Number of samples: " & INTEGER'IMAGE(NUM_SAMPLES) SEVERITY NOTE;

        FOR sample_idx IN 0 TO NUM_SAMPLES - 1 LOOP
            REPORT "Processing sample " & INTEGER'IMAGE(sample_idx) SEVERITY NOTE;

            -- Stream FEATURE_SIZE pixels
            FOR pixel_idx IN 0 TO FEATURE_SIZE - 1 LOOP
                UNIFORM(seed1, seed2, rand_val);
                pixel_val := INTEGER(FLOOR(rand_val * 256.0));

                WAIT UNTIL rising_edge(clk);

                TVALID_M <= '1';
                TDATA_M  <= STD_LOGIC_VECTOR(TO_UNSIGNED(pixel_val, INBIT));
                TKEEP_M  <= "1";

                IF pixel_idx = FEATURE_SIZE - 1 THEN
                    TLAST_M <= '1';
                ELSE
                    TLAST_M <= '0';
                END IF;

                WAIT UNTIL rising_edge(clk) AND TREADY_M = '1';
            END LOOP;

            -- Deassert stream signals
            TVALID_M <= '0';
            TLAST_M  <= '0';
            TDATA_M  <= (OTHERS => '0');

            -- Wait for output to be ready
            IF TVALID_S /= '1' THEN
                WAIT UNTIL rising_edge(clk) AND TVALID_S = '1';
            END IF;

            -- Accept the output
            TREADY_S <= '1';
            WAIT FOR CLK_PERIOD;

            REPORT "  -> Predicted class: " &
                INTEGER'IMAGE(TO_INTEGER(UNSIGNED(TDATA_S(LOG_CLASSES - 1 DOWNTO 0))))
                SEVERITY NOTE;

            -- Small delay between samples
            WAIT FOR CLK_PERIOD * 5;
        END LOOP;

        REPORT "=== LFSR Streaming Simulation Complete ===" SEVERITY NOTE;

        -- End simulation
        WAIT FOR CLK_PERIOD * 20;
        sim_done <= TRUE;
        WAIT;
    END PROCESS;

    ----------------------------------------------------------------------------
    -- Output monitor process
    ----------------------------------------------------------------------------
    output_monitor : PROCESS
        VARIABLE output_count : INTEGER := 0;
    BEGIN
        WHILE NOT sim_done LOOP
            WAIT UNTIL rising_edge(clk);
            IF TVALID_S = '1' AND TREADY_S = '1' THEN
                output_count := output_count + 1;
            END IF;
        END LOOP;
        REPORT "Total outputs received: " & INTEGER'IMAGE(output_count) SEVERITY NOTE;
        WAIT;
    END PROCESS;

END ARCHITECTURE sim;
