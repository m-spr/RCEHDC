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
             lenTKEEP_S   : INTEGER := 1
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
        TKEEP_S  : OUT STD_LOGIC_VECTOR(lenTKEEP_S - 1 DOWNTO 0)
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
            classIndex                 : OUT STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0)
        );
    END COMPONENT OTFGEn;

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

BEGIN
    --rstl <= not(rst);
    HDCOTFGEn: OTFGEn
        GENERIC MAP (
            pixbit, d, lgf, c, featureSize, n, adI, adz, zComp, lgCn, logn, log2features, log2id
        )
        PORT MAP (
            clk, rst, run,
            pixelIn, done, TLAST_S, TVALID_S, TREADY_M,
            classIndex
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
