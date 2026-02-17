LIBRARY IEEE;
    USE IEEE.STD_LOGIC_1164.ALL;
    USE IEEE.NUMERIC_STD.ALL;

ENTITY classifier IS
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
END ENTITY classifier;

ARCHITECTURE behavioral OF classifier IS

    COMPONENT countingSimTop IS
        GENERIC (n           : INTEGER := 10; --; 	-- bit-widths of memory pointer, counter and etc,,, 
                 d           : INTEGER := 10; -- number of confComp module
                 z           : INTEGER := 0;  -- zeropadding to 2** for RSA 
                 classNumber : INTEGER := 10; ---- class number --- for memory image
                 logInNum    : INTEGER := 3); -- MuxCell, ceilingLOG2(#popCounters)
        PORT (
            clk, rst, run : IN  STD_LOGIC;
            hv            : IN  STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
            update_valid  : IN  STD_LOGIC;
            updated_truth : IN  std_logic_vector (999 downto 0);
            updated_prediction : IN std_logic_vector (999 downto 0);
            ground_truth  : IN  integer;
            predicted_label : IN integer;

            done          : OUT STD_LOGIC;
            pointer       : OUT STD_LOGIC_VECTOR(n - 1 DOWNTO 0);
            dout          : OUT STD_LOGIC_VECTOR(classNumber * (n + logInNum) - 1 DOWNTO 0);
            update_done   : OUT STD_LOGIC

        );
    END COMPONENT countingSimTop;

    COMPONENT comparatorTop IS
        GENERIC (len : INTEGER := 8;  -- bit width out adder
                 n   : INTEGER := 10; -- #Classes       ---- all 4s in this code can be replaced with LOG2(n)
                 z   : INTEGER := 10; -- zeropadding to 2**
                 lgn : INTEGER := 4); ---LOG2(n)
        PORT (
            clk, rst, run           : IN  STD_LOGIC;
            a                       : IN  STD_LOGIC_VECTOR(n * len - 1 DOWNTO 0); --- 16 = 2**4 ,,, 4 is LOG2(n)
            ground_truth            : IN integer;
            done, TLAST_S, TVALID_S : OUT STD_LOGIC;                              --- final result is ready 
            classIndex              : OUT STD_LOGIC_VECTOR(lgn - 1 DOWNTO 0);      --- only the index of class can be also the value!  As of now only support up to 16 classes so 4'bits
            predictedClassScore     : OUT STD_LOGIC_VECTOR(len - 1 DOWNTO 0);
            groundTruthScore        : OUT STD_LOGIC_VECTOR(len - 1 DOWNTO 0)
        );
    END COMPONENT comparatorTop;
    SIGNAL hvTOcount : STD_LOGIC_VECTOR(adI - 1 DOWNTO 0);
    SIGNAL dones     : STD_LOGIC;
    SIGNAL toComp    : STD_LOGIC_VECTOR(c * (n + logn) - 1 DOWNTO 0);
    SIGNAL point     : STD_LOGIC_VECTOR(n - 1 DOWNTO 0);
    SIGNAL classIndexI: STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0);

    ATTRIBUTE MARK_DEBUG              : string;
    ATTRIBUTE MARK_DEBUG OF toComp    : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF hvTOcount : SIGNAL IS "TRUE";

BEGIN
    classIndex <= classIndexI;
    
    concat: FOR I IN adI - 1 DOWNTO 0 GENERATE
        hvTOcount(I) <= hv(to_integer(unsigned(point)) + (2 ** n) * I);
    END GENERATE concat;

    CST: countingSimTop
        GENERIC MAP (n, adI, adz, c, logn)
        PORT MAP (
            clk, rst, run,
            hvTOcount,
            update_valid,
            updated_truth,
            updated_prediction,
            ground_truth,
            TO_INTEGER (unsigned (classIndexI)),
            dones,
            point,
            toComp,
            update_done
        );

    CT: comparatorTop
        GENERIC MAP ((n + logn), c, zComp, lgCn)
        PORT MAP (
            clk, rst, dones,
            toComp,
            ground_truth,
            done, TLAST_S, TVALID_S,
            classIndexI,
            predictedClassScore,
            groundTruthScore
         );

    pointer <= point;
END ARCHITECTURE behavioral;
