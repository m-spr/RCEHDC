LIBRARY IEEE;
    USE IEEE.STD_LOGIC_1164.ALL;
    USE IEEE.NUMERIC_STD.ALL;

    USE STD.textio.ALL;
    USE ieee.std_logic_textio.ALL;

ENTITY fulltopHDC IS
    GENERIC (pixbit       : INTEGER := 8;    -- consider 8 bit is enough for grayscale --- it is not
             d            : INTEGER := 1000; -- dimension size
             lgf          : INTEGER := 10;   -- bit width out popCounters --- LOG2(#feature)
             c            : INTEGER := 10;   ---- #Classes
             featureSize  : INTEGER := 784;
             n            : INTEGER := 9;    --512 each classMem -- 2^n <= F, n is max possible number and indicate the bit-widths of memory pointer, counter and etc,,, for comparitor thinpg! 256 unit in each portin of memory
             adI          : INTEGER := 2;    -- number of confComp module, or adderInput and = ceiling(D/(2^n))
             adz          : INTEGER := 0;    -- zeropadding for RSA = 2**? - adI
             zComp        : INTEGER := 6;    -- zeropadding Mux Comp = 2**? - c
             lgCn         : INTEGER := 4;    -- ceilingLOG2(#Classes)
             logn         : INTEGER := 1;    -- MuxCell RSA, ceilingLOG2(#popCounters OR adI)
             log2features : INTEGER := 2;    --log2 of feature size
             log2id       : INTEGER := 1;    --log2 of id level
             lenTKEEP_M   : INTEGER := 1;
             lenTDATA_S   : INTEGER := 8;
             lenTKEEP_S   : INTEGER := 1;
		
		
		C_S00_AXI_Lite_DATA_WIDTH : integer	:= 32;
        C_S00_AXI_Lite_ADDR_WIDTH : integer	:= 5
            );
    PORT (
        clk      : IN  STD_LOGIC;
        rst      : IN  STD_LOGIC;
        TVALID_M : IN  STD_LOGIC;
        TDATA_M  : IN  STD_LOGIC_VECTOR(pixbit - 1 DOWNTO 0);
        TKEEP_M  : IN  STD_LOGIC_VECTOR(lenTKEEP_M - 1 DOWNTO 0);
        TREADY_S : IN  STD_LOGIC;
        TLAST_M  : IN  STD_LOGIC;
        TREADY_M : OUT STD_LOGIC; -- should be always '1' as of now! for DMA only
        TVALID_S : OUT STD_LOGIC;
        TLAST_S  : OUT STD_LOGIC;
        TDATA_S  : OUT STD_LOGIC_VECTOR(lenTDATA_S - 1 DOWNTO 0);
        TKEEP_S  : OUT STD_LOGIC_VECTOR(lenTKEEP_S - 1 DOWNTO 0);
        
        s00_axi_lite_aclk	: in std_logic;
        s00_axi_lite_aresetn	: in std_logic;
        
        s00_axi_lite_awaddr	: in std_logic_vector(C_S00_AXI_Lite_ADDR_WIDTH-1 downto 0);
        s00_axi_lite_awprot	: in std_logic_vector(2 downto 0);
        s00_axi_lite_awvalid	: in std_logic;
        s00_axi_lite_awready	: out std_logic;
        s00_axi_lite_wdata	: in std_logic_vector(C_S00_AXI_Lite_DATA_WIDTH-1 downto 0);
        s00_axi_lite_wstrb	: in std_logic_vector((C_S00_AXI_Lite_DATA_WIDTH/8)-1 downto 0);
        s00_axi_lite_wvalid	: in std_logic;
        s00_axi_lite_wready	: out std_logic;
        s00_axi_lite_bresp	: out std_logic_vector(1 downto 0);
        s00_axi_lite_bvalid	: out std_logic;
        s00_axi_lite_bready	: in std_logic;
        s00_axi_lite_araddr	: in std_logic_vector(C_S00_AXI_Lite_ADDR_WIDTH-1 downto 0);
        s00_axi_lite_arprot	: in std_logic_vector(2 downto 0);
        s00_axi_lite_arvalid	: in std_logic;
        s00_axi_lite_arready	: out std_logic;
        s00_axi_lite_rdata	: out std_logic_vector(C_S00_AXI_Lite_DATA_WIDTH-1 downto 0);
        s00_axi_lite_rresp	: out std_logic_vector(1 downto 0);
        s00_axi_lite_rvalid	: out std_logic;
        s00_axi_lite_rready	: in std_logic
    );
END ENTITY fulltopHDC;

ARCHITECTURE behavioral OF fulltopHDC IS

    COMPONENT OTFGEn IS
        GENERIC (pixbit       : INTEGER := 10;   -- consider 8 bit is enough for grayscale --- it is not
                 d            : INTEGER := 2000; -- dimension size
                 lgf          : INTEGER := 10;   -- bit width out popCounters --- LOG2(#feature)
                 c            : INTEGER := 10;   ---- #Classes
                 featureSize  : INTEGER := 784;
                 n            : INTEGER := 9;    --512 each classMem -- 2^n <= F, n is max possible number and indicate the bit-widths of memory pointer, counter and etc,,, for comparitor thinpg! 256 unit in each portin of memory
                 adI          : INTEGER := 2;    -- number of confComp module, or adderInput and = ceiling(D/(2^n))
                 adz          : INTEGER := 0;    -- zeropadding for RSA = 2**? - adI
                 zComp        : INTEGER := 6;    -- zeropadding Mux Comp = 2**? - c
                 lgCn         : INTEGER := 4;    -- ceilingLOG2(#Classes)
                 logn         : INTEGER := 1;    -- MuxCell RSA, ceilingLOG2(#popCounters OR adI)
                 log2features : INTEGER := 2;    --log2 of feature size
                 log2id       : INTEGER := 1     --log2 of idlevel
                );
        PORT (
            clk                        : IN  STD_LOGIC;
            rstl                       : IN  STD_LOGIC;
            run                        : IN  STD_LOGIC;
            pixel                      : IN  STD_LOGIC_VECTOR(pixbit - 1 DOWNTO 0);
            --update		: IN STD_LOGIC;		
            done                       : OUT STD_LOGIC;
            TLAST_S, TVALID_S, ready_M : OUT STD_LOGIC;
            --pixelMemOutIndex : OUT STD_LOGIC_VECTOR(14 DOWNTO 0);
            classIndex                 : OUT STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0);
            ground_truth               : IN integer;
            learning                   : IN std_logic;
            -- MMIO BRAM access ports
            mmio_active                : IN  STD_LOGIC;
            mmio_bram_sel              : IN  STD_LOGIC_VECTOR(1 DOWNTO 0);
            mmio_addr                  : IN  STD_LOGIC_VECTOR(15 DOWNTO 0);
            mmio_we                    : IN  STD_LOGIC;
            mmio_wdata_wide            : IN  STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
            mmio_wdata_narrow          : IN  STD_LOGIC_VECTOR(31 DOWNTO 0);
            mmio_rdata_wide            : OUT STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
            mmio_rdata_narrow          : OUT STD_LOGIC_VECTOR(31 DOWNTO 0)
        );
    END COMPONENT OTFGEn;

component mmio_handler is
        generic (
            C_S_AXI_DATA_WIDTH	: integer	:= 32;
            C_S_AXI_ADDR_WIDTH	: integer	:= 5
        );
        port (
            S_AXI_ACLK	    : in std_logic;
            S_AXI_ARESETN	: in std_logic;
            S_AXI_AWADDR	: in std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
            S_AXI_AWPROT	: in std_logic_vector(2 downto 0);
            S_AXI_AWVALID	: in std_logic;
            S_AXI_AWREADY	: out std_logic;
            S_AXI_WDATA	    : in std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
            S_AXI_WSTRB	    : in std_logic_vector((C_S_AXI_DATA_WIDTH/8)-1 downto 0);
            S_AXI_WVALID	: in std_logic;
            S_AXI_WREADY	: out std_logic;
            S_AXI_BRESP	    : out std_logic_vector(1 downto 0);
            S_AXI_BVALID	: out std_logic;
            S_AXI_BREADY	: in std_logic;
            S_AXI_ARADDR	: in std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
            S_AXI_ARPROT	: in std_logic_vector(2 downto 0);
            S_AXI_ARVALID	: in std_logic;
            S_AXI_ARREADY	: out std_logic;
            S_AXI_RDATA	    : out std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
            S_AXI_RRESP	    : out std_logic_vector(1 downto 0);
            S_AXI_RVALID	: out std_logic;
            S_AXI_RREADY	: in std_logic;
            reg0_out : OUT std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
            reg1_out : OUT std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
            reg2_out : OUT std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
            reg3_out : OUT std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
            reg4_out : OUT std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
            reg5_out : OUT std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
            bram_rdata_in  : IN  std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
            bram_status_in : IN  std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
            reg5_wr_pulse  : OUT std_logic
        );
    end component mmio_handler;

    COMPONENT regOne IS
        GENERIC (init : STD_LOGIC := '1'); -- initial value
        PORT (
            clk               : IN  STD_LOGIC;
            regUpdate, regrst : IN  STD_LOGIC;
            din               : IN  STD_LOGIC;
            dout              : OUT STD_LOGIC
        );
    END COMPONENT regOne;

    SIGNAL pixelIn    : STD_LOGIC_VECTOR(pixbit - 1 DOWNTO 0);
    SIGNAL classIndex : STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0);

    SIGNAL rstl, run, done : STD_LOGIC;
    SIGNAL outreg0         : std_logic_vector(31 DOWNTO 0) := (OTHERS => '0');
    SIGNAL pixelreg        : STD_LOGIC_VECTOR(pixbit - 1 DOWNTO 0);
    
    SIGNAL ground_truth    : std_logic_vector(31 DOWNTO 0) := (OTHERS => '0');
    SIGNAL reg1_out        : std_logic_vector(C_S00_AXI_Lite_DATA_WIDTH - 1 DOWNTO 0);
    SIGNAL reg2_out        : std_logic_vector(C_S00_AXI_Lite_DATA_WIDTH - 1 DOWNTO 0);
    SIGNAL reg3_out        : std_logic_vector(C_S00_AXI_Lite_DATA_WIDTH - 1 DOWNTO 0);
    SIGNAL reg4_out        : std_logic_vector(C_S00_AXI_Lite_DATA_WIDTH - 1 DOWNTO 0);
    SIGNAL reg5_out        : std_logic_vector(C_S00_AXI_Lite_DATA_WIDTH - 1 DOWNTO 0);
    SIGNAL reg5_wr_pulse   : std_logic;

    TYPE state IS (init, registering);
    SIGNAL ns, ps : state;
    ATTRIBUTE MARK_DEBUG             : string;
    ATTRIBUTE MARK_DEBUG OF TVALID_M : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF TDATA_M  : SIGNAL IS "TRUE";
    --attribute MARK_DEBUG of pixelMemOutIndex : signal is "TRUE";
    ATTRIBUTE MARK_DEBUG OF TREADY_S   : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF TLAST_M    : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF TREADY_M   : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF TVALID_S   : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF TLAST_S    : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF TDATA_S    : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF classIndex : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF done       : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF ns         : SIGNAL IS "TRUE";

    -- MMIO BRAM access control signals (decoded from reg2)
    SIGNAL mmio_active     : std_logic;
    SIGNAL bram_sel        : std_logic_vector(1 DOWNTO 0);
    SIGNAL mmio_we_ctrl    : std_logic;
    SIGNAL mmio_start_raw  : std_logic;

    -- Start edge detection
    SIGNAL start_prev      : std_logic := '0';
    SIGNAL start_pulse     : std_logic := '0';

    -- BRAM access state
    SIGNAL bram_read_pending : std_logic := '0';
    SIGNAL mmio_we_pulse     : std_logic := '0';

    -- 1024-bit capture register for ID/BV wide BRAM reads/writes (d bits used, rest zero-padded)
    SIGNAL capture_wide    : std_logic_vector(1023 DOWNTO 0) := (OTHERS => '0');
    SIGNAL capture_narrow  : std_logic_vector(31 DOWNTO 0) := (OTHERS => '0');
    SIGNAL capture_valid   : std_logic := '0';

    -- Chunk index (0..31) for 32-bit slicing of the 1024-bit capture register
    SIGNAL chunk_idx       : integer range 0 to 31;

    -- OTFGEn MMIO interface
    SIGNAL otf_mmio_rdata_wide   : std_logic_vector(d - 1 DOWNTO 0);
    SIGNAL otf_mmio_rdata_narrow : std_logic_vector(31 DOWNTO 0);

    -- MMIO read data and status fed back to mmio_handler
    SIGNAL bram_rdata_in   : std_logic_vector(31 DOWNTO 0);
    SIGNAL bram_status_in  : std_logic_vector(31 DOWNTO 0);

BEGIN
    --rstl <= not(rst);

    -- Decode MMIO control register (reg2)
    mmio_active    <= reg2_out(0);
    bram_sel       <= reg2_out(2 DOWNTO 1);
    mmio_we_ctrl   <= reg2_out(3);
    mmio_start_raw <= reg2_out(4);

    chunk_idx <= to_integer(unsigned(reg4_out(4 DOWNTO 0)));

    -- Read data mux: select between wide (chunked) and narrow capture
    bram_rdata_in <= capture_wide(chunk_idx * 32 + 31 DOWNTO chunk_idx * 32)
                     WHEN (bram_sel = "00" OR bram_sel = "01") ELSE
                     capture_narrow;

    bram_status_in <= (0 => capture_valid, OTHERS => '0');

    -- Start pulse edge detection and BRAM access sequencing
    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF rst = '0' THEN -- Active-low reset (rst is S_AXI_ARESETN)
                start_prev         <= '0';
                start_pulse        <= '0';
                bram_read_pending  <= '0';
                mmio_we_pulse      <= '0';
                capture_valid      <= '0';
                capture_wide       <= (OTHERS => '0');
                capture_narrow     <= (OTHERS => '0');
            ELSE
                start_prev  <= mmio_start_raw;
                start_pulse <= mmio_start_raw AND (NOT start_prev);
                mmio_we_pulse <= '0';

                -- Handle start pulse: initiate BRAM read or write
                IF (mmio_start_raw = '1' AND start_prev = '0') THEN
                    IF mmio_we_ctrl = '0' THEN
                        bram_read_pending <= '1';
                        capture_valid     <= '0';
                    ELSE
                        mmio_we_pulse <= '1';
                    END IF;
                END IF;

                -- One cycle after read was initiated, capture the data
                IF bram_read_pending = '1' THEN
                    bram_read_pending <= '0';
                    capture_valid     <= '1';
                    IF bram_sel = "10" THEN
                        capture_narrow <= otf_mmio_rdata_narrow;
                    ELSE
                        capture_wide <= (OTHERS => '0');
                        capture_wide(d - 1 DOWNTO 0) <= otf_mmio_rdata_wide;
                    END IF;
                END IF;

                -- Load a 32-bit chunk into capture_wide when PS writes to reg5
                IF reg5_wr_pulse = '1' AND mmio_active = '1' AND bram_sel /= "10" THEN
                    capture_wide(chunk_idx * 32 + 31 DOWNTO chunk_idx * 32) <= reg5_out;
                END IF;
            END IF;
        END IF;
    END PROCESS;

mmio_handler_inst : mmio_handler
    generic map (
        C_S_AXI_DATA_WIDTH	=> C_S00_AXI_Lite_DATA_WIDTH,
        C_S_AXI_ADDR_WIDTH	=> C_S00_AXI_Lite_ADDR_WIDTH
    )
    port map (
        S_AXI_ACLK	    => clk,    
        S_AXI_ARESETN	=> rst,   
        S_AXI_AWADDR	=> s00_axi_lite_awaddr,
        S_AXI_AWPROT	=> s00_axi_lite_awprot,
        S_AXI_AWVALID	=> s00_axi_lite_awvalid,
        S_AXI_AWREADY	=> s00_axi_lite_awready,
        S_AXI_WDATA	    => s00_axi_lite_wdata,
        S_AXI_WSTRB	    => s00_axi_lite_wstrb,
        S_AXI_WVALID	=> s00_axi_lite_wvalid,
        S_AXI_WREADY	=> s00_axi_lite_wready,
        S_AXI_BRESP	    => s00_axi_lite_bresp,
        S_AXI_BVALID	=> s00_axi_lite_bvalid,
        S_AXI_BREADY	=> s00_axi_lite_bready,
        S_AXI_ARADDR	=> s00_axi_lite_araddr,
        S_AXI_ARPROT	=> s00_axi_lite_arprot,
        S_AXI_ARVALID	=> s00_axi_lite_arvalid,
        S_AXI_ARREADY	=> s00_axi_lite_arready,
        S_AXI_RDATA	    => s00_axi_lite_rdata,
        S_AXI_RRESP	    => s00_axi_lite_rresp,
        S_AXI_RVALID	=> s00_axi_lite_rvalid,
        S_AXI_RREADY	=> s00_axi_lite_rready,
        reg0_out        => ground_truth,
        reg1_out        => reg1_out,
        reg2_out        => reg2_out,
        reg3_out        => reg3_out,
        reg4_out        => reg4_out,
        reg5_out        => reg5_out,
        bram_rdata_in   => bram_rdata_in,
        bram_status_in  => bram_status_in,
        reg5_wr_pulse   => reg5_wr_pulse
    );
    HDCOTFGEn: OTFGEn
        GENERIC MAP (
            pixbit, d, lgf, c, featureSize, n, adI, adz, zComp, lgCn, logn, log2features, log2id
        )
        PORT MAP (
            clk            => clk,
            rstl           => rst,
            run            => run,
            pixel          => pixelIn,
            done           => done,
            TLAST_S        => TLAST_S,
            TVALID_S       => TVALID_S,
            ready_M        => TREADY_M,
            classIndex     => classIndex,
            ground_truth   => TO_INTEGER(unsigned(ground_truth)),
            learning       => reg1_out(0),
            mmio_active    => mmio_active,
            mmio_bram_sel  => bram_sel,
            mmio_addr      => reg3_out(15 DOWNTO 0),
            mmio_we        => mmio_we_pulse,
            mmio_wdata_wide   => capture_wide(d - 1 DOWNTO 0),
            mmio_wdata_narrow => reg5_out,
            mmio_rdata_wide   => otf_mmio_rdata_wide,
            mmio_rdata_narrow => otf_mmio_rdata_narrow
        );

    pixelIn <= TDATA_M;
    run     <= TVALID_M;
    --TREADY_M <= not(TLAST_M);
    ---TREADY_M <= '1';
    TDATA_S <= "0000" & classIndex;
    TKEEP_S <= "1";

    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF (rst = '1') THEN
                ps <= init;
            ELSE
                ps <= ns;
            END IF;
        END IF;
    END PROCESS;

    --	PROCESS ( ps,  done, TREADY_S)
    --	BEGIN 
    --	TLAST_S <= '0';
    --    TVALID_S <= '0';
    --		CASE (ps) IS 
    --			WHEN init =>
    --                IF ( done = '1') THEN
    --                 --TLAST_S <= '1';
    --                 --TVALID_S <= '1';
    --                    ns <= registering;
    --                Else
    --                    ns <= init;
    --                END IF;
    --            --ns <= registering;
    --			WHEN registering =>
    --                TLAST_S <= '1';
    --                TVALID_S <= '1';
    --                IF (TREADY_S = '1') THEN  --- perhaps -1 is extra! check
    --                    ns <= init;
    --				ELSE
    --					ns <= registering;
    --				END IF;
    --			WHEN OTHERS =>
    --					ns <= init;
    --		END CASE;
    --	END PROCESS;
    --    regTLAST_S : regOne 
    --	GENERIC MAP('0')
    --	PORT MAP(
    --		clk , done, rst, done, TLAST_S  
    --	);
    --    regTVALID_S : regOne 
    --	GENERIC MAP('0')
    --	PORT MAP(
    --		clk , done, rst, done, TVALID_S  
    --	);
END ARCHITECTURE behavioral;
