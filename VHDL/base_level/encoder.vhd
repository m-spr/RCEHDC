LIBRARY IEEE;
    USE IEEE.STD_LOGIC_1164.ALL;
    USE IEEE.NUMERIC_STD.ALL;

ENTITY encoder IS
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
END ENTITY encoder;

ARCHITECTURE behavioral OF encoder IS

    COMPONENT reg IS
        GENERIC (lenPop : INTEGER := 8); -- bit width out popCounters
        PORT (
            clk               : IN  STD_LOGIC;
            regUpdate, regrst : IN  STD_LOGIC;
            din               : IN  STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0);
            dout              : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
        );
    END COMPONENT reg;

    COMPONENT XoringInputPop IS
        GENERIC (n : INTEGER := 8 -- number of loop
                );
        PORT (
            clk, rst, update : IN  STD_LOGIC; --- bit width out popCounters --- LOG2(#feature)
            done             : IN  STD_LOGIC;
            din              : IN  STD_LOGIC;
            BV               : IN  STD_LOGIC;
            dout             : OUT STD_LOGIC_VECTOR(n - 1 DOWNTO 0)
        );
    END COMPONENT XoringInputPop;

    COMPONENT XoringPopCtrl IS
        GENERIC (n           : INTEGER := 10; ----bit width out popCounters --- ceiling log2(#feature)
                 featureSize : INTEGER := 700); ---- NUmber of loops for popcounts 
        PORT (
            clk, rst                                : IN  STD_LOGIC;
            run                                     : IN  STD_LOGIC;
            counter                                 : OUT STD_LOGIC_VECTOR(n - 1 DOWNTO 0);
            rundegi, update, doneI, doneII, ready_M : OUT STD_LOGIC
        );
    END COMPONENT XoringPopCtrl;

    SIGNAL update, doneI : STD_LOGIC;
    CONSTANT test : STD_LOGIC_VECTOR(lgf - 1 DOWNTO 0) := STD_LOGIC_VECTOR(to_UNSIGNED(featureSize / 2, lgf));
    SIGNAL douttest       : STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
    SIGNAL doutXOR        : STD_LOGIC_VECTOR((d * lgf) - 1 DOWNTO 0);
    SIGNAL querycheck     : STD_LOGIC_VECTOR((d) - 1 DOWNTO 0);
    SIGNAL testID         : STD_LOGIC_VECTOR(1000 - 1 DOWNTO 0) := (OTHERS => '0');
    SIGNAL testBV         : STD_LOGIC_VECTOR(1000 - 1 DOWNTO 0) := (OTHERS => '0');
    SIGNAL IDeq, BVeq, eq : STD_LOGIC;

    ATTRIBUTE MARK_DEBUG            : string;
    ATTRIBUTE MARK_DEBUG OF doutXOR : SIGNAL IS "TRUE";

BEGIN

    popCounters: FOR I IN 0 TO d - 1 GENERATE
        pop: XoringInputPop
            GENERIC MAP (lgf)
            PORT MAP (
                clk, rst, update, doneI,
                din(I), BV(I),
                doutXOR((I + 1) * lgf - 1 DOWNTO (I) * lgf)
            );
    END GENERATE popCounters;

    ctrl: XoringPopCtrl
        GENERIC MAP (lgf, featureSize)
        PORT MAP (
            clk, rst, run, counter, rundegi,
            update, doneI, done, ready_M
        );

    doutGen: FOR I IN 0 TO d - 1 GENERATE
        dout(I) <= '1' WHEN (doutXOR((I + 1) * lgf - 1 DOWNTO (I) * lgf) > test) ELSE '0';
    END GENERATE doutGen;

END ARCHITECTURE behavioral;
