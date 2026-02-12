--------------------------------------------------------------------------------
-- Testbench: tb_encoder_stream
--
-- Streams a single 784-pixel sample into fulltopHDC using AXI-Stream.
-- The pixel values are read from a text file (one integer per line).
-- No output comparison is performed; use the Python reference output instead.
--------------------------------------------------------------------------------

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
USE STD.TEXTIO.ALL;

ENTITY tb_encoder_stream IS
END ENTITY tb_encoder_stream;

ARCHITECTURE sim OF tb_encoder_stream IS

    CONSTANT CLK_PERIOD   : TIME    := 20 ns;
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

    CONSTANT PIXEL_FILE : STRING := "/home/jakob/Documents/GitHub/RCEHDC/mnist_example/encoder_pixels.txt";

    SIGNAL clk      : STD_LOGIC := '0';
    SIGNAL rst      : STD_LOGIC := '0';

    SIGNAL TVALID_M : STD_LOGIC := '0';
    SIGNAL TDATA_M  : STD_LOGIC_VECTOR(PIXBIT - 1 DOWNTO 0) := (OTHERS => '0');
    SIGNAL TKEEP_M  : STD_LOGIC_VECTOR(0 DOWNTO 0) := "1";
    SIGNAL TREADY_M : STD_LOGIC;
    SIGNAL TLAST_M  : STD_LOGIC := '0';

    SIGNAL TVALID_S : STD_LOGIC;
    SIGNAL TDATA_S  : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL TKEEP_S  : STD_LOGIC_VECTOR(0 DOWNTO 0);
    SIGNAL TREADY_S : STD_LOGIC := '1';
    SIGNAL TLAST_S  : STD_LOGIC;

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

    SIGNAL sim_done : BOOLEAN := FALSE;

BEGIN

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

    s00_axi_lite_aclk <= clk;

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

    streamer : PROCESS
        FILE pixel_f : TEXT OPEN READ_MODE IS PIXEL_FILE;
        VARIABLE ln : LINE;
        VARIABLE pixel_val : INTEGER;
        VARIABLE pixel_idx : INTEGER;
    BEGIN
        rst <= '0';
        s00_axi_lite_aresetn <= '0';
        WAIT FOR CLK_PERIOD * 10;
        rst <= '1';
        s00_axi_lite_aresetn <= '1';
        WAIT FOR CLK_PERIOD * 10;

        FOR pixel_idx IN 0 TO FEATURE_SIZE - 1 LOOP
            IF endfile(pixel_f) THEN
                REPORT "Pixel file ended early" SEVERITY FAILURE;
            END IF;

            readline(pixel_f, ln);
            read(ln, pixel_val);

            TVALID_M <= '1';
            TDATA_M  <= STD_LOGIC_VECTOR(TO_UNSIGNED(pixel_val, PIXBIT));

            IF pixel_idx = FEATURE_SIZE - 1 THEN
                TLAST_M <= '1';
            ELSE
                TLAST_M <= '0';
            END IF;

            LOOP
                WAIT UNTIL rising_edge(clk);
                IF TREADY_M = '1' THEN
                    EXIT;
                END IF;
            END LOOP;
        END LOOP;

        TVALID_M <= '0';
        TLAST_M  <= '0';
        TDATA_M  <= (OTHERS => '0');

        WAIT FOR CLK_PERIOD * 20;
        sim_done <= TRUE;
        WAIT;
    END PROCESS;

END ARCHITECTURE sim;
