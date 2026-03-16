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

ENTITY OTFGEn IS
    GENERIC (
        inbit       : INTEGER := 10;  -- Input bit-width (grayscale)
        d           : INTEGER := 2000; -- Dimension size
        lgf         : INTEGER := 10;  -- LOG2(featureSize)
        c           : INTEGER := 10;  -- Number of classes
        featureSize : INTEGER := 784;
        n           : INTEGER := 9;  -- Max memory pointer bit-width
        adI         : INTEGER := 2;  -- Number of confComp modules
        adz         : INTEGER := 0;  -- Zero padding for RSA
        zComp       : INTEGER := 6;  -- Zero padding for MUX Comparator
        lgCn        : INTEGER := 4;  -- CeilingLOG2(#Classes)
        logn        : INTEGER := 1;  -- MuxCell RSA
        r           : INTEGER := 2;  -- Remainder for ID level
        x           : INTEGER := 1   -- Coefficient of IDLEVEL
    );
    PORT (
        clk        : IN  STD_LOGIC;
        rstl       : IN  STD_LOGIC;
        run        : IN  STD_LOGIC;
        pixel      : IN  STD_LOGIC_VECTOR(inbit-1 DOWNTO 0);
        done       : OUT STD_LOGIC;
        TLAST_S    : OUT STD_LOGIC;
        TVALID_S   : OUT STD_LOGIC;
        ready_M    : OUT STD_LOGIC;
        classIndex : OUT STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0);
        ground_truth : IN  INTEGER;
        learning     : IN  std_logic
    );
END ENTITY OTFGEn;

ARCHITECTURE behavioral OF OTFGEn IS

    SIGNAL rst : STD_LOGIC;

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
        
    COMPONENT popCount IS
        GENERIC (
            lenPop : INTEGER := 8  -- Bit width of popCounters
        );
        PORT (
            clk : IN  STD_LOGIC;
            rst : IN  STD_LOGIC;
            en  : IN  STD_LOGIC;
            dout : OUT STD_LOGIC_VECTOR(lenPop-1 DOWNTO 0)
        );
    END COMPONENT;

    COMPONENT BasedVectorLFSR IS
        GENERIC (
            n : INTEGER := 2000  -- Number of bits
        );
        PORT (
            clk    : IN  STD_LOGIC;
            rst    : IN  STD_LOGIC;
            update : IN  STD_LOGIC;
            dout   : OUT STD_LOGIC_VECTOR(n-1 DOWNTO 0)
        );
    END COMPONENT;

    COMPONENT idLevel3 IS
        GENERIC (
            n  : INTEGER := 7;  -- Input bit-width
            c  : INTEGER := 2;  -- Increment coefficient
            r  : INTEGER := 2;  -- ID-level remainder
            hv : INTEGER := 500  -- Hyperdimensional size
        );
        PORT (
            values   : IN  STD_LOGIC_VECTOR(n-1 DOWNTO 0);
            idVector : OUT STD_LOGIC_VECTOR(hv-1 DOWNTO 0)
        );
    END COMPONENT;

    COMPONENT encoder IS
        GENERIC (
            d           : INTEGER := 500;  -- Dimension size
            lgf         : INTEGER := 10;   -- LOG2(#feature)
            featureSize : INTEGER := 700
        );
        PORT (
            clk      : IN  STD_LOGIC;
            rst      : IN  STD_LOGIC;
            run      : IN  STD_LOGIC;
            din      : IN  STD_LOGIC_VECTOR(d-1 DOWNTO 0);
            BV       : IN  STD_LOGIC_VECTOR(d-1 DOWNTO 0);
            rundegi  : OUT STD_LOGIC;
            done     : OUT STD_LOGIC;
            ready_M  : OUT STD_LOGIC;
            dout     : OUT STD_LOGIC_VECTOR(d-1 DOWNTO 0)
        );
    END COMPONENT;

    COMPONENT classifier IS
        GENERIC (d     : INTEGER := 1000; --- dimension size+zeropading
                 dimensionSize : INTEGER := 1000; -- actual dimension size
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
            updated_truth           : IN  STD_LOGIC_VECTOR(dimensionSize - 1 DOWNTO 0);
            updated_prediction      : IN  STD_LOGIC_VECTOR(dimensionSize - 1 DOWNTO 0);
            ground_truth            : IN  INTEGER;
            update_valid            : IN  STD_LOGIC;
            done, TLAST_S, TVALID_S : OUT STD_LOGIC;
            pointer                 : OUT STD_LOGIC_VECTOR(n - 1 DOWNTO 0);
            classIndex              : OUT STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0);
            predictedClassScore     : OUT STD_LOGIC_VECTOR((n + logn) - 1 DOWNTO 0);
            groundTruthScore        : OUT STD_LOGIC_VECTOR((n + logn) - 1 DOWNTO 0);
            update_done             : OUT STD_LOGIC
            );
    END COMPONENT;

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
            done                 : OUT STD_LOGIC
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

    -- Signals
    SIGNAL doneEncoderToClassifier, rundegi, popen, rstpop1, rstpop, bvrst : STD_LOGIC;
    SIGNAL QHV, query_checker, idLevelOut, idLevelOutreg : STD_LOGIC_VECTOR(d-1 DOWNTO 0);
    SIGNAL QHV_reg : STD_LOGIC_VECTOR(d-1 DOWNTO 0) := (others => '0');
    SIGNAL encoderTodiv : STD_LOGIC_VECTOR(adI*(2**n) - 1 DOWNTO 0);
    SIGNAL BV : STD_LOGIC_VECTOR(d-1 DOWNTO 0);
    SIGNAL divToClass : STD_LOGIC_VECTOR(adI - 1 DOWNTO 0);
    SIGNAL pointer : STD_LOGIC_VECTOR(n-1 DOWNTO 0);
    SIGNAL classIndexI : STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0);
    SIGNAL indexdatamem : STD_LOGIC_VECTOR(14 DOWNTO 0);
    SIGNAL indexdatamem11 : STD_LOGIC_VECTOR(12 DOWNTO 0);

    CONSTANT encodeVecZero : STD_LOGIC_VECTOR(adI*(2**n)-d-1 DOWNTO 0) := (others => '0');

    -- Learning flow signals
    SIGNAL binary_correct       : STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
    SIGNAL binary_predicted     : STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
    SIGNAL predictedClassScore  : STD_LOGIC_VECTOR((n + logn) - 1 DOWNTO 0);
    SIGNAL groundTruthScore     : STD_LOGIC_VECTOR((n + logn) - 1 DOWNTO 0);
    SIGNAL learningRun          : STD_LOGIC := '0';
    SIGNAL learning_done        : STD_LOGIC;
    SIGNAL update_done          : STD_LOGIC;
    SIGNAL TLAST_SI, TVALID_SI  : STD_LOGIC;
    SIGNAL doneEncoderToClassifier_d : STD_LOGIC := '0';

BEGIN

    classIndex <= classIndexI;
    rst <= NOT rstl;
    encoderTodiv <= encodeVecZero & QHV_reg;
    rstpop1 <= '1' WHEN indexdatamem11 = "1010101110000" ELSE '0';
    rstpop <= rstpop1 OR rst;
    indexdatamem <= indexdatamem11 & "00";

    -- Pop Count Component
    pop : popCount
        GENERIC MAP (
            lenPop => 13
        )
        PORT MAP (
            clk  => clk,
            rst  => rstpop,
            en   => rundegi,
            dout => indexdatamem11
        );

    bvrst <= rst OR doneEncoderToClassifier OR rstpop;
    
    -- ID Level Generator
    idGen : idLevel3
        GENERIC MAP (
            n  => inbit,
            c  => x,
            r  => r,
            hv => d
        )
        PORT MAP (
            values  => pixel,
            idVector => idLevelOut
        );

    -- Based Vector Generator
    BVGen : BasedVectorLFSR
        GENERIC MAP (
            n => d
        )
        PORT MAP (
            clk    => clk,
            rst    => bvrst,
            update => rundegi,
            dout   => BV
        );

    -- Encoder
    enc : encoder
        GENERIC MAP (
            d           => d,
            lgf         => lgf,
            featureSize => featureSize
        )
        PORT MAP (
            clk      => clk,
            rst      => rst,
            run      => run,
            din      => idLevelOut,
            BV       => BV,
            rundegi  => rundegi,
            done     => doneEncoderToClassifier,
            ready_M  => ready_M,
            dout     => QHV
        );

    -- Classifier
    cls : classifier
        GENERIC MAP (
            d     => adI*(2**n),
            dimensionSize => d,
            c     => c,
            n     => n, 
            adI   => adI,
            adz   => adz,
            zComp => zComp,
            lgCn  => lgCn,
            logn  => logn
        )
        PORT MAP (
            clk                => clk,
            rst                => rst,
            run                => doneEncoderToClassifier_d,
            hv                 => encoderTodiv,
            updated_truth      => binary_correct,
            updated_prediction => binary_predicted,
            ground_truth       => ground_truth,
            update_valid       => learning_done,
            done               => done,
            TLAST_S            => TLAST_SI,
            TVALID_S           => TVALID_SI,
            pointer            => pointer,
            classIndex         => classIndexI,
            predictedClassScore => predictedClassScore,
            groundTruthScore    => groundTruthScore,
            update_done         => update_done
        );

    -- Learning Top
    learn_inst : learningTop
        GENERIC MAP (d, c)
        PORT MAP (
            clk, rst, learningRun,
            ground_truth,
            to_integer(unsigned(classIndexI)),
            to_integer(unsigned(groundTruthScore)),
            to_integer(unsigned(predictedClassScore)),
            QHV_reg,
            binary_correct,
            binary_predicted,
            learning_done
        );

    -- Learning Flow Controller
    learning_ctrl : learningFlowCtrl
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

    -- Delay process for encoder-to-classifier handshake and QHV registration
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
