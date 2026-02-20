-- MIT License

-- Copyright (c) 2024 m-spr

-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:

-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.

-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
USE STD.TEXTIO.ALL;
USE IEEE.STD_LOGIC_TEXTIO.ALL;

ENTITY fulltopHDC IS
    GENERIC (
        inbit                  : INTEGER := 8;   -- Number of bits for input
        dimension              : INTEGER := 1000; -- HDC Dimension size
        pruning                : INTEGER := 336; -- Number of efficient dimensions
        logfeature             : INTEGER := 10; -- LOG2(featureSize)
        classes                : INTEGER := 10; -- Number of classes
        featureSize            : INTEGER := 784; -- Number of elements in each input data
        classMemSize           : INTEGER := 7;  -- Length of each segment of classHyper memories
        confCompNum            : INTEGER := 3;  -- Number of confComp modules in the comparator
        rsaZeropadding         : INTEGER := 1;  -- Zero paddings for the sequential adder (RSA)
        comparatorZeroPadding  : INTEGER := 6;  -- Zero paddings for multiplexer in comparators
        logClasses             : INTEGER := 4;  -- ceiling[LOG2(classes)]
        logn                   : INTEGER := 2;  -- ceilingLOG2(#popCounters OR adI)
        IDreminder             : INTEGER := 232; -- Remainder value for ID-level
        IDcoefficient          : INTEGER := 3;  -- Coefficient of ID-level
        lenTKEEP_M             : INTEGER := 1;
        lenTDATA_S             : INTEGER := 8;
        lenTKEEP_S             : INTEGER := 1;

        C_S00_AXI_Lite_DATA_WIDTH : INTEGER := 32;
        C_S00_AXI_Lite_ADDR_WIDTH : INTEGER := 4
    );
    PORT (
        clk        : IN  STD_LOGIC; 
        rst        : IN  STD_LOGIC; 
        TVALID_M   : IN  STD_LOGIC;         
        TDATA_M    : IN  STD_LOGIC_VECTOR(inbit-1 DOWNTO 0);
        TKEEP_M    : IN  STD_LOGIC_VECTOR(lenTKEEP_M-1 DOWNTO 0);
        TREADY_S   : IN  STD_LOGIC;   
        TLAST_M    : IN  STD_LOGIC;    
        TREADY_M   : OUT STD_LOGIC;  -- Should always be '1' for DMA only
        TVALID_S   : OUT STD_LOGIC;         
        TLAST_S    : OUT STD_LOGIC;         
        TDATA_S    : OUT STD_LOGIC_VECTOR(lenTDATA_S-1 DOWNTO 0);
        TKEEP_S    : OUT STD_LOGIC_VECTOR(lenTKEEP_S-1 DOWNTO 0);

        s00_axi_lite_aclk    : IN  std_logic;
        s00_axi_lite_aresetn : IN  std_logic;
        s00_axi_lite_awaddr  : IN  std_logic_vector(C_S00_AXI_Lite_ADDR_WIDTH-1 downto 0);
        s00_axi_lite_awprot  : IN  std_logic_vector(2 downto 0);
        s00_axi_lite_awvalid : IN  std_logic;
        s00_axi_lite_awready : OUT std_logic;
        s00_axi_lite_wdata   : IN  std_logic_vector(C_S00_AXI_Lite_DATA_WIDTH-1 downto 0);
        s00_axi_lite_wstrb   : IN  std_logic_vector((C_S00_AXI_Lite_DATA_WIDTH/8)-1 downto 0);
        s00_axi_lite_wvalid  : IN  std_logic;
        s00_axi_lite_wready  : OUT std_logic;
        s00_axi_lite_bresp   : OUT std_logic_vector(1 downto 0);
        s00_axi_lite_bvalid  : OUT std_logic;
        s00_axi_lite_bready  : IN  std_logic;
        s00_axi_lite_araddr  : IN  std_logic_vector(C_S00_AXI_Lite_ADDR_WIDTH-1 downto 0);
        s00_axi_lite_arprot  : IN  std_logic_vector(2 downto 0);
        s00_axi_lite_arvalid : IN  std_logic;
        s00_axi_lite_arready : OUT std_logic;
        s00_axi_lite_rdata   : OUT std_logic_vector(C_S00_AXI_Lite_DATA_WIDTH-1 downto 0);
        s00_axi_lite_rresp   : OUT std_logic_vector(1 downto 0);
        s00_axi_lite_rvalid  : OUT std_logic;
        s00_axi_lite_rready  : IN  std_logic
    );
END ENTITY fulltopHDC;

ARCHITECTURE behavioral OF fulltopHDC IS

    COMPONENT OTFGEn IS
        GENERIC (
            inbit     : INTEGER := 10; 
            d         : INTEGER := 2000; -- Dimension size
            lgf       : INTEGER := 10; -- LOG2(#feature)
            c         : INTEGER := 10; -- Number of classes
            featureSize : INTEGER := 784;
            n         : INTEGER := 9;  -- Memory pointer bit-widths
            adI       : INTEGER := 2;  -- Number of confComp modules
            adz       : INTEGER := 0;  -- Zero padding for RSA
            zComp     : INTEGER := 6;  -- Zero padding for MUX Comparator
            lgCn      : INTEGER := 4;  -- CeilingLOG2(#Classes)
            logn      : INTEGER := 1;  -- MuxCell RSA
            r         : INTEGER := 2;  -- Remainder for ID-level
            x         : INTEGER := 1   -- Coefficient of IDLEVEL
        );
        PORT (
            clk       : IN  STD_LOGIC; 
            rstl      : IN  STD_LOGIC; 
            run       : IN  STD_LOGIC;
            pixel     : IN  STD_LOGIC_VECTOR(inbit-1 DOWNTO 0);
            done      : OUT STD_LOGIC;
            TLAST_S   : OUT STD_LOGIC;
            TVALID_S  : OUT STD_LOGIC;
            ready_M   : OUT STD_LOGIC;
            classIndex : OUT STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0);
            ground_truth : IN INTEGER;
            learning     : IN std_logic
        );
    END COMPONENT;

    COMPONENT mmio_handler IS
        GENERIC (
            C_S_AXI_DATA_WIDTH : INTEGER := 32;
            C_S_AXI_ADDR_WIDTH : INTEGER := 4
        );
        PORT (
            S_AXI_ACLK    : IN  std_logic;
            S_AXI_ARESETN : IN  std_logic;
            S_AXI_AWADDR  : IN  std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
            S_AXI_AWPROT  : IN  std_logic_vector(2 downto 0);
            S_AXI_AWVALID : IN  std_logic;
            S_AXI_AWREADY : OUT std_logic;
            S_AXI_WDATA   : IN  std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
            S_AXI_WSTRB   : IN  std_logic_vector((C_S_AXI_DATA_WIDTH/8)-1 downto 0);
            S_AXI_WVALID  : IN  std_logic;
            S_AXI_WREADY  : OUT std_logic;
            S_AXI_BRESP   : OUT std_logic_vector(1 downto 0);
            S_AXI_BVALID  : OUT std_logic;
            S_AXI_BREADY  : IN  std_logic;
            S_AXI_ARADDR  : IN  std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
            S_AXI_ARPROT  : IN  std_logic_vector(2 downto 0);
            S_AXI_ARVALID : IN  std_logic;
            S_AXI_ARREADY : OUT std_logic;
            S_AXI_RDATA   : OUT std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
            S_AXI_RRESP   : OUT std_logic_vector(1 downto 0);
            S_AXI_RVALID  : OUT std_logic;
            S_AXI_RREADY  : IN  std_logic;
            reg0_out : OUT std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
            reg1_out : OUT std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0)
        );
    END COMPONENT mmio_handler;

    COMPONENT regOne IS
        GENERIC (
            init : STD_LOGIC := '1'  -- Initial value
        );
        PORT (
            clk       : IN  STD_LOGIC;
            regUpdate : IN  STD_LOGIC;
            regrst    : IN  STD_LOGIC;
            din       : IN  STD_LOGIC;
            dout      : OUT STD_LOGIC
        );
    END COMPONENT;
        
    SIGNAL pixelIn     : STD_LOGIC_VECTOR(inbit-1 DOWNTO 0);
    SIGNAL classIndex  : STD_LOGIC_VECTOR(logClasses - 1 DOWNTO 0);
    SIGNAL rstl, run, done : STD_LOGIC;
    SIGNAL outreg0     : STD_LOGIC_VECTOR(31 DOWNTO 0) := (others => '0');
    SIGNAL pixelreg    : STD_LOGIC_VECTOR(inbit-1 DOWNTO 0);

    SIGNAL ground_truth : std_logic_vector(31 DOWNTO 0) := (OTHERS => '0');
    SIGNAL reg1_out     : std_logic_vector(C_S00_AXI_Lite_DATA_WIDTH - 1 DOWNTO 0);

    CONSTANT ALLZERO : STD_LOGIC_VECTOR(lenTDATA_S-logClasses-1 DOWNTO 0) := (others => '0');

    TYPE state IS (init, registering);
    SIGNAL ns, ps : state;

BEGIN

    -- MMIO Handler for AXI-Lite register access
    mmio_handler_inst : mmio_handler
        GENERIC MAP (
            C_S_AXI_DATA_WIDTH => C_S00_AXI_Lite_DATA_WIDTH,
            C_S_AXI_ADDR_WIDTH => C_S00_AXI_Lite_ADDR_WIDTH
        )
        PORT MAP (
            S_AXI_ACLK    => clk,
            S_AXI_ARESETN => rst,
            S_AXI_AWADDR  => s00_axi_lite_awaddr,
            S_AXI_AWPROT  => s00_axi_lite_awprot,
            S_AXI_AWVALID => s00_axi_lite_awvalid,
            S_AXI_AWREADY => s00_axi_lite_awready,
            S_AXI_WDATA   => s00_axi_lite_wdata,
            S_AXI_WSTRB   => s00_axi_lite_wstrb,
            S_AXI_WVALID  => s00_axi_lite_wvalid,
            S_AXI_WREADY  => s00_axi_lite_wready,
            S_AXI_BRESP   => s00_axi_lite_bresp,
            S_AXI_BVALID  => s00_axi_lite_bvalid,
            S_AXI_BREADY  => s00_axi_lite_bready,
            S_AXI_ARADDR  => s00_axi_lite_araddr,
            S_AXI_ARPROT  => s00_axi_lite_arprot,
            S_AXI_ARVALID => s00_axi_lite_arvalid,
            S_AXI_ARREADY => s00_axi_lite_arready,
            S_AXI_RDATA   => s00_axi_lite_rdata,
            S_AXI_RRESP   => s00_axi_lite_rresp,
            S_AXI_RVALID  => s00_axi_lite_rvalid,
            S_AXI_RREADY  => s00_axi_lite_rready,
            reg0_out      => ground_truth,
            reg1_out      => reg1_out
        );

    -- Instantiating HDC OTFGEn
    HDCOTFGEn : OTFGEn 
        GENERIC MAP (
            inbit      => inbit, 
            d          => dimension, 
            lgf        => logfeature, 
            c          => classes, 
            featureSize => featureSize, 
            n          => classMemSize, 
            adI        => confCompNum, 
            adz        => rsaZeropadding, 
            zComp      => comparatorZeroPadding, 
            lgCn       => logClasses, 
            logn       => logn, 
            r          => IDreminder, 
            x          => IDcoefficient
        )
        PORT MAP (
            clk       => clk, 
            rstl      => rst, 
            run       => run, 
            pixel     => pixelIn, 
            done      => done,  
            TLAST_S   => TLAST_S, 
            TVALID_S  => TVALID_S, 
            ready_M   => TREADY_M,  
            classIndex => classIndex,
            ground_truth => TO_INTEGER(unsigned(ground_truth)),
            learning     => reg1_out(0)
        );

    -- Assignments
    pixelIn <= TDATA_M;
    run     <= TVALID_M; 
    TDATA_S <= ALLZERO & classIndex;
    TKEEP_S <= (others => '1');

    -- State Machine
    PROCESS(clk) 
    BEGIN 
        IF rising_edge(clk) THEN
            IF (rst = '0') THEN
                ps <= init; 
            ELSE  
                ps <= ns;  
            END IF;
        END IF;
    END PROCESS;

END ARCHITECTURE behavioral;
