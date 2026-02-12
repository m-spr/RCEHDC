--------------------------------------------------------------------------------
-- Testbench: tb_online_training
-- 
-- Simulates the online training behavior from the PYNQ notebook:
--   1. Enable learning mode via AXI-Lite write to reg1 (address 0x04)
--   2. Stream 784 pixels per sample via AXI-Stream (DMA send channel)
--   3. Write ground truth label via AXI-Lite to reg0 (address 0x00)
--   4. Receive classification output via AXI-Stream (DMA recv channel)
--   5. Repeat for NUM_SAMPLES training samples
--   6. Disable learning mode
--
-- This testbench generates pseudo-random training data to simulate MNIST-like
-- behavior without requiring actual image files.
--
-- NOTE: In the updated fulltop.vhd, the mmio_handler uses the main clock (clk)
-- and reset (rst) directly, not the s00_axi_lite_aclk/aresetn ports. The reset
-- signal 'rst' is connected to S_AXI_ARESETN which expects active-LOW logic.
--------------------------------------------------------------------------------

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
USE IEEE.MATH_REAL.ALL;

ENTITY tb_online_training IS
END ENTITY tb_online_training;

ARCHITECTURE sim OF tb_online_training IS

    -- Constants matching the HDC design parameters
    CONSTANT CLK_PERIOD   : TIME    := 20 ns;  -- 50 MHz clock
    CONSTANT PIXBIT       : INTEGER := 8;
    CONSTANT D            : INTEGER := 1000;
    CONSTANT LGF          : INTEGER := 10;
    CONSTANT C            : INTEGER := 10;
    CONSTANT FEATURE_SIZE : INTEGER := 784;
    CONSTANT N            : INTEGER := 9;
    CONSTANT ADI          : INTEGER := 2;
    CONSTANT ADZ          : INTEGER := 0;
    CONSTANT ZCOMP        : INTEGER := 6;
    CONSTANT LGCN         : INTEGER := 4;
    CONSTANT LOGN         : INTEGER := 1;
    CONSTANT LOG2FEATURES : INTEGER := 2;
    CONSTANT LOG2ID       : INTEGER := 1;
    
    CONSTANT C_AXI_DATA_WIDTH : INTEGER := 32;
    CONSTANT C_AXI_ADDR_WIDTH : INTEGER := 4;
    
    -- Number of training samples to simulate
    CONSTANT NUM_SAMPLES : INTEGER := 20;
    
    -- DUT signals
    SIGNAL clk      : STD_LOGIC := '0';
    SIGNAL rst      : STD_LOGIC := '0';
    
    -- AXI-Stream Master interface (DMA send -> HDC)
    SIGNAL TVALID_M : STD_LOGIC := '0';
    SIGNAL TDATA_M  : STD_LOGIC_VECTOR(PIXBIT - 1 DOWNTO 0) := (OTHERS => '0');
    SIGNAL TKEEP_M  : STD_LOGIC_VECTOR(0 DOWNTO 0) := "1";
    SIGNAL TREADY_M : STD_LOGIC;
    SIGNAL TLAST_M  : STD_LOGIC := '0';
    
    -- AXI-Stream Slave interface (HDC -> DMA recv)
    SIGNAL TVALID_S : STD_LOGIC;
    SIGNAL TDATA_S  : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL TKEEP_S  : STD_LOGIC_VECTOR(0 DOWNTO 0);
    SIGNAL TREADY_S : STD_LOGIC := '1';
    SIGNAL TLAST_S  : STD_LOGIC;
    
    -- AXI-Lite interface signals
    SIGNAL s00_axi_lite_aclk     : STD_LOGIC := '0';
    SIGNAL s00_axi_lite_aresetn  : STD_LOGIC := '0';
    SIGNAL s00_axi_lite_awaddr   : STD_LOGIC_VECTOR(C_AXI_ADDR_WIDTH - 1 DOWNTO 0) := (OTHERS => '0');
    SIGNAL s00_axi_lite_awprot   : STD_LOGIC_VECTOR(2 DOWNTO 0) := (OTHERS => '0');
    SIGNAL s00_axi_lite_awvalid  : STD_LOGIC := '0';
    SIGNAL s00_axi_lite_awready  : STD_LOGIC;
    SIGNAL s00_axi_lite_wdata    : STD_LOGIC_VECTOR(C_AXI_DATA_WIDTH - 1 DOWNTO 0) := (OTHERS => '0');
    SIGNAL s00_axi_lite_wstrb    : STD_LOGIC_VECTOR((C_AXI_DATA_WIDTH / 8) - 1 DOWNTO 0) := (OTHERS => '1');
    SIGNAL s00_axi_lite_wvalid   : STD_LOGIC := '0';
    SIGNAL s00_axi_lite_wready   : STD_LOGIC;
    SIGNAL s00_axi_lite_bresp    : STD_LOGIC_VECTOR(1 DOWNTO 0);
    SIGNAL s00_axi_lite_bvalid   : STD_LOGIC;
    SIGNAL s00_axi_lite_bready   : STD_LOGIC := '1';
    SIGNAL s00_axi_lite_araddr   : STD_LOGIC_VECTOR(C_AXI_ADDR_WIDTH - 1 DOWNTO 0) := (OTHERS => '0');
    SIGNAL s00_axi_lite_arprot   : STD_LOGIC_VECTOR(2 DOWNTO 0) := (OTHERS => '0');
    SIGNAL s00_axi_lite_arvalid  : STD_LOGIC := '0';
    SIGNAL s00_axi_lite_arready  : STD_LOGIC;
    SIGNAL s00_axi_lite_rdata    : STD_LOGIC_VECTOR(C_AXI_DATA_WIDTH - 1 DOWNTO 0);
    SIGNAL s00_axi_lite_rresp    : STD_LOGIC_VECTOR(1 DOWNTO 0);
    SIGNAL s00_axi_lite_rvalid   : STD_LOGIC;
    SIGNAL s00_axi_lite_rready   : STD_LOGIC := '1';
    
    -- Simulation control
    SIGNAL sim_done : BOOLEAN := FALSE;
    
    -- For pseudo-random data generation
    SHARED VARIABLE seed1 : POSITIVE := 1;
    SHARED VARIABLE seed2 : POSITIVE := 2;

BEGIN

    -- Clock generation
    -- NOTE: mmio_handler internally uses 'clk' directly, not s00_axi_lite_aclk
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
    
    -- AXI-Lite clock follows main clock (port exists but not used internally)
    s00_axi_lite_aclk <= clk;

    -- Device Under Test instantiation
    DUT : ENTITY work.fulltopHDC
        GENERIC MAP (
            pixbit               => PIXBIT,
            d                    => D,
            lgf                  => LGF,
            c                    => C,
            featureSize          => FEATURE_SIZE,
            n                    => N,
            adI                  => ADI,
            adz                  => ADZ,
            zComp                => ZCOMP,
            lgCn                 => LGCN,
            logn                 => LOGN,
            log2features         => LOG2FEATURES,
            log2id               => LOG2ID,
            lenTKEEP_M           => 1,
            lenTDATA_S           => 8,
            lenTKEEP_S           => 1,
            C_S00_AXI_Lite_DATA_WIDTH => C_AXI_DATA_WIDTH,
            C_S00_AXI_Lite_ADDR_WIDTH => C_AXI_ADDR_WIDTH
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
            TKEEP_S  => TKEEP_S,
            s00_axi_lite_aclk    => s00_axi_lite_aclk,
            s00_axi_lite_aresetn => s00_axi_lite_aresetn,
            s00_axi_lite_awaddr  => s00_axi_lite_awaddr,
            s00_axi_lite_awprot  => s00_axi_lite_awprot,
            s00_axi_lite_awvalid => s00_axi_lite_awvalid,
            s00_axi_lite_awready => s00_axi_lite_awready,
            s00_axi_lite_wdata   => s00_axi_lite_wdata,
            s00_axi_lite_wstrb   => s00_axi_lite_wstrb,
            s00_axi_lite_wvalid  => s00_axi_lite_wvalid,
            s00_axi_lite_wready  => s00_axi_lite_wready,
            s00_axi_lite_bresp   => s00_axi_lite_bresp,
            s00_axi_lite_bvalid  => s00_axi_lite_bvalid,
            s00_axi_lite_bready  => s00_axi_lite_bready,
            s00_axi_lite_araddr  => s00_axi_lite_araddr,
            s00_axi_lite_arprot  => s00_axi_lite_arprot,
            s00_axi_lite_arvalid => s00_axi_lite_arvalid,
            s00_axi_lite_arready => s00_axi_lite_arready,
            s00_axi_lite_rdata   => s00_axi_lite_rdata,
            s00_axi_lite_rresp   => s00_axi_lite_rresp,
            s00_axi_lite_rvalid  => s00_axi_lite_rvalid,
            s00_axi_lite_rready  => s00_axi_lite_rready
        );

    ----------------------------------------------------------------------------
    -- AXI-Lite Write Procedure
    -- Simulates: hdc.fulltopHDC_0.write(addr, data)
    ----------------------------------------------------------------------------
    axi_lite_write : PROCESS
        PROCEDURE do_axi_lite_write(
            CONSTANT addr : IN STD_LOGIC_VECTOR(C_AXI_ADDR_WIDTH - 1 DOWNTO 0);
            CONSTANT data : IN STD_LOGIC_VECTOR(C_AXI_DATA_WIDTH - 1 DOWNTO 0)
        ) IS
        BEGIN
            -- Set address and data
            s00_axi_lite_awaddr  <= addr;
            s00_axi_lite_wdata   <= data;
            s00_axi_lite_awvalid <= '1';
            s00_axi_lite_wvalid  <= '1';
            
            -- Wait for ready signals
            WAIT UNTIL rising_edge(clk) AND s00_axi_lite_awready = '1' AND s00_axi_lite_wready = '1';
            
            -- Deassert valid signals
            s00_axi_lite_awvalid <= '0';
            s00_axi_lite_wvalid  <= '0';
            
            -- Wait for write response
            WAIT UNTIL rising_edge(clk) AND s00_axi_lite_bvalid = '1';
            
            -- Small delay between transactions
            WAIT FOR CLK_PERIOD * 2;
        END PROCEDURE;
        
        VARIABLE rand_val   : REAL;
        VARIABLE pixel_val  : INTEGER;
        VARIABLE label_val  : INTEGER;
        VARIABLE sample_idx : INTEGER;
        
    BEGIN
        -- Initial reset sequence
        -- NOTE: In fulltop.vhd, mmio_handler's S_AXI_ARESETN is connected to 'rst'
        -- S_AXI_ARESETN is active-LOW, so rst='0' means mmio_handler is in reset
        -- and rst='1' means mmio_handler is out of reset (normal operation)
        -- However, the FSM uses rst='1' as reset. This creates a timing mismatch.
        -- For proper operation: rst should be '1' briefly, then go to '0' for reset,
        -- but the FSM expects rst='1' for reset. We use rst='1' during init.
        rst <= '0';  -- Assert reset (FSM in reset, mmio_handler NOT in reset)
        s00_axi_lite_aresetn <= '0';  -- Not used internally, but drive it anyway
        WAIT FOR CLK_PERIOD * 10;
        rst <= '1';  -- Deassert reset (FSM normal, mmio_handler in reset - design quirk)
        s00_axi_lite_aresetn <= '1';  -- Not used internally
        WAIT FOR CLK_PERIOD * 10;
        
        REPORT "=== Starting Online Training Simulation ===" SEVERITY NOTE;
        REPORT "Number of samples: " & INTEGER'IMAGE(NUM_SAMPLES) SEVERITY NOTE;
        
        ------------------------------------------------------------------------
        -- Enable learning mode: hdc.fulltopHDC_0.write(0x04, 1)
        ------------------------------------------------------------------------
        REPORT "Enabling learning mode (write 1 to 0x04)" SEVERITY NOTE;
        do_axi_lite_write(X"4", X"00000001");
        
        ------------------------------------------------------------------------
        -- Training loop: Process NUM_SAMPLES samples
        ------------------------------------------------------------------------
        FOR sample_idx IN 0 TO NUM_SAMPLES - 1 LOOP
            -- Generate pseudo-random label (0-9)
            UNIFORM(seed1, seed2, rand_val);
            label_val := INTEGER(FLOOR(rand_val * 10.0));
            
            REPORT "Processing sample " & INTEGER'IMAGE(sample_idx) & 
                   " with label " & INTEGER'IMAGE(label_val) SEVERITY NOTE;

            do_axi_lite_write(X"0", STD_LOGIC_VECTOR(TO_UNSIGNED(label_val, C_AXI_DATA_WIDTH)));
            
            -- Stream 784 pixels (Simulating high-speed DMA burst)
            for pixel_idx in 0 to FEATURE_SIZE - 1 loop
                -- 1. Prepare data for the current cycle
                UNIFORM(seed1, seed2, rand_val);
                pixel_val := INTEGER(FLOOR(rand_val * 256.0));
            
                -- 2. Drive the AXI-Stream bus
                TVALID_M <= '1';
                TDATA_M  <= STD_LOGIC_VECTOR(TO_UNSIGNED(pixel_val, PIXBIT));
                
                if pixel_idx = FEATURE_SIZE - 1 then
                    TLAST_M <= '1';
                else
                    TLAST_M <= '0';
                end if;
            
                -- 3. The Handshake: Wait for the rising edge where TREADY is '1'
                -- If the PL is ready, this loop exits in exactly 1 CLK_PERIOD
                loop
                    wait until rising_edge(clk);
                    if TREADY_M = '1' then
                        exit; 
                    end if;
                end loop;
            end loop;

            -- 4. Clean up: Deassert after the last pixel is accepted
            TVALID_M <= '0';
            TLAST_M  <= '0';
            -- Deassert stream signals
            TVALID_M <= '0';
            TLAST_M  <= '0';
            TDATA_M  <= (OTHERS => '0');
            
            -- Wait for output to be ready (classification done)
            IF TVALID_S /= '1' THEN
                WAIT UNTIL rising_edge(clk) AND TVALID_S = '1';
            END IF;
            
            -- Accept the output
            TREADY_S <= '1';
            WAIT FOR CLK_PERIOD;
            
            -- Log the predicted class
            REPORT "  -> Predicted class: " & INTEGER'IMAGE(TO_INTEGER(UNSIGNED(TDATA_S(3 DOWNTO 0)))) SEVERITY NOTE;
            
            WAIT FOR CLK_PERIOD * 5;
        END LOOP;
        
        ------------------------------------------------------------------------
        -- Disable learning mode: hdc.fulltopHDC_0.write(0x04, 0)
        ------------------------------------------------------------------------
        REPORT "Disabling learning mode (write 0 to 0x04)" SEVERITY NOTE;
        do_axi_lite_write(X"4", X"00000000");
        
        REPORT "=== Online Training Simulation Complete ===" SEVERITY NOTE;
        REPORT "Processed " & INTEGER'IMAGE(NUM_SAMPLES) & " training samples" SEVERITY NOTE;
        
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
