LIBRARY IEEE;
    USE IEEE.STD_LOGIC_1164.ALL;
    USE IEEE.NUMERIC_STD.ALL;

    USE STD.textio.ALL;
    USE ieee.std_logic_textio.ALL;

ENTITY OTFGEn IS
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
        ground_truth               : IN  INTEGER;
        learning                   : IN  std_logic;
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
END ENTITY OTFGEn;

ARCHITECTURE behavioral OF OTFGEn IS

    SIGNAL rst : std_logic;

    COMPONENT regOne IS
        GENERIC (init : STD_LOGIC := '1'); -- initial value
        PORT (
            clk               : IN  STD_LOGIC;
            regUpdate, regrst : IN  STD_LOGIC;
            din               : IN  STD_LOGIC;
            dout              : OUT STD_LOGIC
        );
    END COMPONENT regOne;

    COMPONENT popCount IS
        GENERIC (lenPop : INTEGER := 8); -- bit width out popCounters
        PORT (
            clk, rst : IN  STD_LOGIC;
            en       : IN  STD_LOGIC;
            dout     : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
        );
    END COMPONENT popCount;

    COMPONENT blk_mem_gen_BV IS
        PORT (
            clka  : IN  STD_LOGIC;
            wea   : IN  STD_LOGIC_VECTOR(0 DOWNTO 0);
            addra : IN  STD_LOGIC_VECTOR(lgf - 1 DOWNTO 0);
            dina  : IN  STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
            douta : OUT STD_LOGIC_VECTOR(d - 1 DOWNTO 0)
        );
    END COMPONENT blk_mem_gen_BV;

    COMPONENT blk_mem_gen_ID IS
        PORT (
            clka  : IN  STD_LOGIC;
            wea   : IN  STD_LOGIC_VECTOR(0 DOWNTO 0);
            addra : IN  STD_LOGIC_VECTOR(pixbit - 1 DOWNTO 0);
            dina  : IN  STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
            douta : OUT STD_LOGIC_VECTOR(d - 1 DOWNTO 0)
        );
    END COMPONENT blk_mem_gen_ID;

    COMPONENT encoder IS
        GENERIC (d           : INTEGER := 500; -- dimension size 
                 lgf         : INTEGER := 10;  -- bit width out popCounters --- LOG2(#feature) --- nabayad in bashe!?! whats wrong with me?
                 featureSize : INTEGER := 700
                );
        PORT (
            clk, rst      : IN  STD_LOGIC; -- run should be '1' for 2 clk cycle!!!!! badan behehesh fekr mikonam!!
            run           : IN  STD_LOGIC;
            din           : IN  STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
            BV            : IN  STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
            rundegi       : OUT STD_LOGIC;
            done, ready_M : OUT STD_LOGIC;
            counter       : OUT STD_LOGIC_VECTOR(lgf - 1 DOWNTO 0);
            dout          : OUT STD_LOGIC_VECTOR(d - 1 DOWNTO 0)
        );
    END COMPONENT encoder;

    COMPONENT classifier IS
        GENERIC (d     : INTEGER := 1000; --- dimension size+zeropading
                 c     : INTEGER := 10;   ---- #Classes
                 n     : INTEGER := 7;    -- 2^n <= F, n is max possible number and indicate the bit-widths of memory pointer, counter and etc,,,
                 adI   : INTEGER := 5;    -- number of confComp module, or adderInput and = ceiling(D/(2^n))
                 adz   : INTEGER := 3;    -- zeropadding for RSA = 2**? - adI
                 zComp : INTEGER := 6;    -- zeropadding Mux Comp = 2**? - c
                 lgCn  : INTEGER := 4;    -- ceilingLOG2(#Classes)
                 logn  : INTEGER := 3); -- MuxCell RSA, ceilingLOG2(#popCounters)
        PORT (
            clk, rst, run           : IN  STD_LOGIC;
            hv                      : IN  STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
            updated_truth           : IN  STD_LOGIC_VECTOR(999 DOWNTO 0);
            updated_prediction      : IN  STD_LOGIC_VECTOR(999 DOWNTO 0);
            ground_truth            : IN  INTEGER;
            update_valid            : IN  STD_LOGIC;
            done, TLAST_S, TVALID_S : OUT STD_LOGIC;
            pointer                 : OUT STD_LOGIC_VECTOR(n - 1 DOWNTO 0);
            classIndex              : OUT STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0);
            predictedClassScore     : OUT STD_LOGIC_VECTOR((n + logn) - 1 DOWNTO 0);
            groundTruthScore        : OUT STD_LOGIC_VECTOR((n + logn) - 1 DOWNTO 0);
            update_done             : OUT STD_LOGIC
            );
    END COMPONENT classifier;

    COMPONENT hvTOcompIn IS
        GENERIC (
            d   : INTEGER := 100000; -- dimension size
            n   : INTEGER := 7;      -- 2^n <= F, n is max possible number and indicate the bit-widths of memory pointer, counter and etc,,,
            adI : INTEGER := 20      -- number of confComp module, or adderInput and = ceiling(D/(2^n))
        );
        PORT (
            clk     : IN  STD_LOGIC;
            rst     : IN  STD_LOGIC;
            din     : IN  STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
            pointer : IN  STD_LOGIC_VECTOR(n - 1 DOWNTO 0);
            dout    : OUT STD_LOGIC_VECTOR(adI - 1 DOWNTO 0)
        );
    END COMPONENT hvTOcompIn;

    COMPONENT reg IS
        GENERIC (lenPop : INTEGER := 8); -- bit width out popCounters
        PORT (
            clk               : IN  STD_LOGIC;
            regUpdate, regrst : IN  STD_LOGIC;
            din               : IN  STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0);
            dout              : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
        );
    END COMPONENT reg;

    COMPONENT learningTop IS
        GENERIC (
            d           : INTEGER := 1000;
            num_classes : INTEGER := 10
        );
        PORT (
            clk                  : IN  STD_LOGIC;
            rst                  : IN  STD_LOGIC;
            run                  : IN  STD_LOGIC;
            correct_label        : IN  INTEGER;                          --ground truth label
            predicted_label      : IN  INTEGER;                          --predicted label
            similarity_correct   : IN  INTEGER;                          --hamming distance to correct class vector
            similarity_incorrect : IN  INTEGER;                          --hamming distance to predicted class vector
            qhv                  : IN  STD_LOGIC_VECTOR(d - 1 DOWNTO 0); --wrongly predicted query vector
            binary_correct       : OUT STD_LOGIC_VECTOR(d - 1 DOWNTO 0); --binarized updated correct class weights
            binary_predicted     : OUT STD_LOGIC_VECTOR(d - 1 DOWNTO 0); --binarized updated predicted class weights
            done                 : OUT STD_LOGIC;
            mmio_active          : IN  STD_LOGIC;
            mmio_learn_addr      : IN  STD_LOGIC_VECTOR(15 DOWNTO 0);
            mmio_learn_din       : IN  STD_LOGIC_VECTOR(31 DOWNTO 0);
            mmio_learn_we        : IN  STD_LOGIC;
            mmio_learn_dout      : OUT STD_LOGIC_VECTOR(31 DOWNTO 0)
        );
    END COMPONENT learningTop;

    COMPONENT learningFlowCtrl IS
        GENERIC (lgCn : INTEGER := 4);
        PORT (
            clk, rst      : IN  STD_LOGIC;
            learning      : IN  STD_LOGIC;
            TVALID_SI     : IN  STD_LOGIC;
            TLAST_SI      : IN  STD_LOGIC;
            classIndexI   : IN  STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0);
            ground_truth  : IN  INTEGER;
            update_done   : IN  STD_LOGIC;
            learningRun   : OUT STD_LOGIC;
            TLAST_S       : OUT STD_LOGIC;
            TVALID_S      : OUT STD_LOGIC
        );
    END COMPONENT learningFlowCtrl;

    SIGNAL doneEncoderToClassifier, rundegi, popen, rstpop1, rstpop : STD_LOGIC;
    SIGNAL QHV                                                      : std_logic_vector(d - 1 DOWNTO 0);
    SIGNAL QHV_reg                                                  : std_logic_vector(d - 1 DOWNTO 0) := (others => '0');
    SIGNAL query_checker                                            : std_logic_vector(d - 1 DOWNTO 0);
    SIGNAL encoderTodiv                                             : std_logic_vector(adI * (2 ** n) - 1 DOWNTO 0);
    SIGNAL idLevelOut                                               : std_logic_vector(d - 1 DOWNTO 0);
    SIGNAL idLevelOutreg                                            : std_logic_vector(d - 1 DOWNTO 0);

    SIGNAL pixelreg : STD_LOGIC_VECTOR(pixbit - 1 DOWNTO 0);

    SIGNAL BV          : std_logic_vector(d - 1 DOWNTO 0);
    SIGNAL divToClass  : std_logic_vector(adI - 1 DOWNTO 0);
    SIGNAL pointer     : STD_LOGIC_VECTOR(n - 1 DOWNTO 0);
    SIGNAL counter     : STD_LOGIC_VECTOR(lgf - 1 DOWNTO 0);
    SIGNAL classIndexCorrect : STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0);
    CONSTANT encodeVecZero : STD_LOGIC_VECTOR(adI * (2 ** n) - d - 1 DOWNTO 0) := (OTHERS => '0');
    SIGNAL indexdatamem   : STD_LOGIC_VECTOR(14 DOWNTO 0);
    SIGNAL indexdatamem11 : STD_LOGIC_VECTOR(12 DOWNTO 0);

    SIGNAL binary_correct       : std_logic_vector(999 DOWNTO 0); --binarized updated correct class weights
    SIGNAL binary_predicted     : std_logic_vector(999 DOWNTO 0); --binarized updated predicted class weights
    SIGNAL predictedClassScore  : STD_LOGIC_VECTOR((n + logn) - 1 DOWNTO 0);
    SIGNAL groundTruthScore     : STD_LOGIC_VECTOR((n + logn) - 1 DOWNTO 0);
    SIGNAL learningRun          : STD_LOGIC := '0';
    SIGNAL learning_done        : STD_LOGIC;
    SIGNAL write_done           : std_logic;
    SIGNAL update_done          : STD_LOGIC;
    SIGNAL TLAST_SI, TVALID_SI  : STD_LOGIC;
    SIGNAL classIndexI          : STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0);

    FILE file_VECTORS : text;
    SIGNAL bvrst : std_logic;

    ATTRIBUTE MARK_DEBUG                            : string;
    ATTRIBUTE MARK_DEBUG OF pixel                   : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF BV                      : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF idLevelOut              : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF rstpop                  : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF classIndex              : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF indexdatamem11          : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF QHV                     : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF doneEncoderToClassifier : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF pointer                 : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF counter                 : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF done                    : SIGNAL IS "TRUE";

    SIGNAL doneEncoderToClassifier_d : STD_LOGIC := '0';

    -- MMIO BRAM muxing signals
    SIGNAL id_addr_muxed : STD_LOGIC_VECTOR(pixbit - 1 DOWNTO 0);
    SIGNAL bv_addr_muxed : STD_LOGIC_VECTOR(lgf - 1 DOWNTO 0);
    SIGNAL id_wea        : STD_LOGIC_VECTOR(0 DOWNTO 0);
    SIGNAL bv_wea        : STD_LOGIC_VECTOR(0 DOWNTO 0);

    SIGNAL mmio_learn_active : STD_LOGIC;

BEGIN
    classIndex <= classIndexI;
    rst          <= NOT(rstl);
    encoderTodiv <= encodeVecZero & QHV_reg;
    rstpop1      <= '1' WHEN indexdatamem11 = "1010101110000" ELSE '0';
    rstpop       <= rstpop1 OR rst;
    indexdatamem <= indexdatamem11 & "00";

    -- MMIO address muxing for ID and BV BRAMs
    id_addr_muxed <= mmio_addr(pixbit - 1 DOWNTO 0)
                     WHEN (mmio_active = '1' AND mmio_bram_sel = "00") ELSE pixel;
    bv_addr_muxed <= mmio_addr(lgf - 1 DOWNTO 0)
                     WHEN (mmio_active = '1' AND mmio_bram_sel = "01") ELSE counter;

    id_wea(0) <= mmio_we WHEN (mmio_active = '1' AND mmio_bram_sel = "00") ELSE '0';
    bv_wea(0) <= mmio_we WHEN (mmio_active = '1' AND mmio_bram_sel = "01") ELSE '0';

    mmio_rdata_wide <= idLevelOut WHEN mmio_bram_sel = "00" ELSE BV;

    mmio_learn_active <= '1' WHEN (mmio_active = '1' AND mmio_bram_sel = "10") ELSE '0';

    pop: popCount
        GENERIC MAP (13)
        PORT MAP (
            clk, rstpop,
            rundegi,
            indexdatamem11
        );
    bvrst <= rst OR doneEncoderToClassifier OR rstpop;

    idGen: blk_mem_gen_ID
        PORT MAP (
            clka  => clk,
            wea   => id_wea,
            addra => id_addr_muxed,
            dina  => mmio_wdata_wide,
            douta => idLevelOut
        );

    BVGen: blk_mem_gen_BV
        PORT MAP (
            clka  => clk,
            wea   => bv_wea,
            addra => bv_addr_muxed,
            dina  => mmio_wdata_wide,
            douta => BV
        );

    enc: encoder
        GENERIC MAP (
            d, lgf, featureSize
        )
        PORT MAP (clk, rst, run,
                  idLevelOut, BV, rundegi,
                  doneEncoderToClassifier, ready_M,
                  counter, QHV
        );



    cls: classifier
        GENERIC MAP (adI * (2 ** n), c, n, adI, adz, zComp, lgCn, logn)
        PORT MAP (
            clk, rst, doneEncoderToClassifier_d,  
            encoderTodiv,  
            binary_correct,
            binary_predicted,
            ground_truth,
            learning_done,
            done, TLAST_SI, TVALID_SI, pointer,
            classIndexI,
            predictedClassScore,
            groundTruthScore,
            update_done
        );
        
    learn_inst: learningTop
        GENERIC MAP (d, c)
        PORT MAP (
            clk              => clk,
            rst              => rst,
            run              => learningRun,
            correct_label    => ground_truth,
            predicted_label  => to_integer(unsigned(classIndexI)),
            similarity_correct   => to_integer(unsigned(groundTruthScore)),
            similarity_incorrect => to_integer(unsigned(predictedClassScore)),
            qhv              => QHV_reg,
            binary_correct   => binary_correct,
            binary_predicted => binary_predicted,
            done             => learning_done,
            mmio_active      => mmio_learn_active,
            mmio_learn_addr  => mmio_addr,
            mmio_learn_din   => mmio_wdata_narrow,
            mmio_learn_we    => mmio_we,
            mmio_learn_dout  => mmio_rdata_narrow
        );

    learning_ctrl: learningFlowCtrl
        GENERIC MAP (lgCn)
        PORT MAP (
            clk, rst,
            learning,
            TVALID_SI,
            TLAST_SI,
            classIndexI,
            ground_truth,
            update_done,
            learningRun,
            TLAST_S,
            TVALID_S
        );

    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF rst = '1' THEN
                doneEncoderToClassifier_d <= '0';
                QHV_reg <= (others => '0');
            ELSE
                doneEncoderToClassifier_d <= doneEncoderToClassifier;
                QHV_reg <= QHV;
            END IF;
        END IF;
    END PROCESS;

END ARCHITECTURE behavioral;
